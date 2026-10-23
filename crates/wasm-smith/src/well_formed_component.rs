//! Generation of well-formed WebAssembly components from the component model's
//! [binary grammar].
//!
//! Each method emits the production it is named after. The fuzzer input picks
//! between alternatives and decides how many times a `vec(...)` or `section*`
//! repeats, so every component is a derivation of `component`. Size and count
//! prefixes are written once the bytes they cover exist, and the grammar's side
//! conditions, including the text format's grammar for names, are generated
//! directly.
//!
//! The grammar alone would make nearly every index out of bounds, so each
//! scope (component, component type, instance type and core module type)
//! tracks the length of each sort's index space as definitions are emitted.
//! Indices are picked within those lengths, and productions that need an index
//! aren't picked when there is none. Names are reused from a pool so that
//! imports, exports, aliases and instantiation arguments tend to match. Types
//! aren't tracked.
//!
//! Generation also follows the validation rules that `wasmparser` checks while
//! decoding, so that decoding never fails, and a few cheap validation rules to
//! get further into validation. Every loop stops once the input is exhausted
//! and the first alternative of each choice doesn't recurse, so generation
//! terminates.
//!
//! [binary grammar]: https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md

use crate::core::{
    DuplicateImportsBehavior, arbitrary_memtype, arbitrary_table_type, configured_valtypes,
};
use crate::{Config, Module, arbitrary_loop, limited_str};
use arbitrary::{Arbitrary, Result, Unstructured};
use std::collections::HashSet;
use std::mem;
use wasm_encoder::{Encode, GlobalType, HeapType, RefType, ValType};

/// A pseudo-random well-formed WebAssembly [component].
///
/// The component is a derivation of the `component` production of the
/// component model's [binary grammar], so it always decodes. Indices refer to
/// an existing definition of the right sort when there is one, but types are
/// not tracked, so the component is not necessarily valid.
///
/// Productions that `wasmparser` doesn't decode as specified are not
/// generated: 🪙 value definitions and `valuebound`s, and the `sh?` immediate
/// of the 🧵② built-ins.
///
/// [component]: https://github.com/WebAssembly/component-model/blob/main/design/mvp/Explainer.md
/// [binary grammar]: https://github.com/WebAssembly/component-model/blob/main/design/mvp/Binary.md
#[derive(Debug)]
pub struct WellFormedComponent {
    bytes: Vec<u8>,
}

impl<'a> Arbitrary<'a> for WellFormedComponent {
    fn arbitrary(u: &mut Unstructured<'a>) -> Result<Self> {
        WellFormedComponent::new(Config::default(), u)
    }
}

impl WellFormedComponent {
    /// Generates a component.
    ///
    /// Nesting is bounded by [`Config::max_nesting_depth`], the number of
    /// nested core modules and components by [`Config::max_modules`] and
    /// [`Config::max_components`], and the length of each section by the
    /// corresponding `max_*` field. Core modules are generated with `config`.
    pub fn new(config: Config, u: &mut Unstructured) -> Result<Self> {
        let mut builder = Builder {
            core_valtypes: configured_valtypes(&config),
            config,
            scopes: Vec::new(),
            names: Vec::new(),
            core_names: Vec::new(),
            modules: 0,
            components: 0,
        };
        let mut bytes = Vec::new();
        builder.component(u, &mut bytes)?;
        Ok(WellFormedComponent { bytes })
    }

    /// Returns the encoded component.
    pub fn to_bytes(&self) -> Vec<u8> {
        self.bytes.clone()
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Sort {
    CoreFunc,
    CoreTable,
    CoreMemory,
    CoreGlobal,
    CoreTag,
    CoreType,
    CoreModule,
    CoreInstance,
    Func,
    Value,
    Type,
    Component,
    Instance,
}

const SORTS: usize = 13;

const ALL_SORTS: &[Sort] = &[
    Sort::CoreFunc,
    Sort::CoreTable,
    Sort::CoreMemory,
    Sort::CoreGlobal,
    Sort::CoreTag,
    Sort::CoreType,
    Sort::CoreModule,
    Sort::CoreInstance,
    Sort::Func,
    Sort::Value,
    Sort::Type,
    Sort::Component,
    Sort::Instance,
];

// The grammar allows any sort in a `sortidx` and an `export` alias, and any
// core sort in a `core:sortidx` and a `core export` alias, but `wasmparser`
// rejects the sorts that validation would reject while decoding.
const EXTERN_SORTS: &[Sort] = &[
    Sort::CoreModule,
    Sort::Func,
    Sort::Value,
    Sort::Type,
    Sort::Component,
    Sort::Instance,
];
const CORE_EXTERN_SORTS: &[Sort] = &[
    Sort::CoreFunc,
    Sort::CoreTable,
    Sort::CoreMemory,
    Sort::CoreGlobal,
    Sort::CoreTag,
];

const OUTER_ALIAS_SORTS: &[Sort] = &[
    Sort::CoreModule,
    Sort::CoreType,
    Sort::Component,
    Sort::Type,
];

// Aliases in component and instance types may only refer to these.
const DECL_EXPORT_ALIAS_SORTS: &[Sort] = &[Sort::Instance, Sort::Type];
const DECL_OUTER_ALIAS_SORTS: &[Sort] = &[Sort::CoreType, Sort::Type];

impl Sort {
    fn core(self) -> Option<u8> {
        Some(match self {
            Sort::CoreFunc => 0x00,
            Sort::CoreTable => 0x01,
            Sort::CoreMemory => 0x02,
            Sort::CoreGlobal => 0x03,
            Sort::CoreTag => 0x04,
            Sort::CoreType => 0x10,
            Sort::CoreModule => 0x11,
            Sort::CoreInstance => 0x12,
            _ => return None,
        })
    }

    fn encode(self, out: &mut Vec<u8>) {
        match self.core() {
            Some(sort) => out.extend([0x00, sort]),
            None => out.push(match self {
                Sort::Func => 0x01,
                Sort::Value => 0x02,
                Sort::Type => 0x03,
                Sort::Component => 0x04,
                Sort::Instance => 0x05,
                _ => unreachable!(),
            }),
        }
    }
}

/// A component, component type, instance type or core module type.
#[derive(Default)]
struct Scope {
    /// Whether this is a component or instance type.
    decl: bool,
    /// The length of each sort's index space.
    lens: [u32; SORTS],
    /// The canonical forms of the import and export names so far.
    imports: HashSet<String>,
    exports: HashSet<String>,
    /// Whether this component has a start section, which `wasmparser` allows
    /// only one of while decoding.
    start: bool,
}

/// The `canonopt`s that validation allows on a canonical definition.
#[derive(Clone, Copy, PartialEq)]
enum Opts {
    Lift,
    Lower,
    Sync,
}

/// The extern names that are valid in some position.
#[derive(Clone, Copy, PartialEq)]
enum Names {
    All,
    Unannotated,
    Labels,
}

struct Builder {
    config: Config,
    core_valtypes: Vec<ValType>,
    /// Enclosing scopes, innermost last.
    scopes: Vec<Scope>,
    /// Extern names generated so far, reused so that names line up across
    /// definitions.
    names: Vec<String>,
    /// Names imported or exported by generated core modules.
    core_names: Vec<String>,
    modules: usize,
    components: usize,
}

/// Whether to keep repeating something, which is false once `u` is exhausted
/// and otherwise false with probability `1/n`.
fn more(u: &mut Unstructured, n: u8) -> Result<bool> {
    Ok(u.arbitrary::<u8>()? % n != 0)
}

impl Builder {
    fn scope(&mut self) -> &mut Scope {
        self.scopes.last_mut().unwrap()
    }

    fn len(&self, sort: Sort) -> u32 {
        self.scopes.last().unwrap().lens[sort as usize]
    }

    fn has(&self, sort: Sort) -> bool {
        self.len(sort) > 0
    }

    fn has_any(&self, sorts: &[Sort]) -> bool {
        sorts.iter().any(|s| self.has(*s))
    }

    fn add(&mut self, sort: Sort, n: u32) {
        let len = &mut self.scope().lens[sort as usize];
        *len = len.saturating_add(n);
    }

    fn can_nest(&self) -> bool {
        self.scopes.len() <= self.config.max_nesting_depth
    }

    /// Returns an index into the index space of `sort` in the scope `ct` scopes
    /// out, which is out of bounds only when that index space is empty.
    fn idx(&self, u: &mut Unstructured, sort: Sort, ct: usize) -> Result<u32> {
        let len = self.scopes[self.scopes.len() - 1 - ct].lens[sort as usize];
        match len.checked_sub(1) {
            Some(max) => u.int_in_range(0..=max),
            None => Ok(0),
        }
    }

    fn index(&self, u: &mut Unstructured, out: &mut Vec<u8>, sort: Sort) -> Result<()> {
        let idx = self.idx(u, sort, 0)?;
        self.u32(out, idx);
        Ok(())
    }

    /// Picks one of `sorts`, preferring those with a non-empty index space.
    fn pick_sort(&self, u: &mut Unstructured, sorts: &[Sort]) -> Result<Sort> {
        let nonempty: Vec<Sort> = sorts.iter().copied().filter(|s| self.has(*s)).collect();
        Ok(*u.choose(if nonempty.is_empty() {
            sorts
        } else {
            &nonempty
        })?)
    }

    /// A LEB128 `u32`, padded to [`Config::min_uleb_size`] bytes.
    fn u32(&self, out: &mut Vec<u8>, mut value: u32) {
        let min = usize::from(self.config.min_uleb_size).min(5);
        let mut len = 0;
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            len += 1;
            let more = value != 0 || len < min;
            if more {
                byte |= 0x80;
            }
            out.push(byte);
            if !more {
                break;
            }
        }
    }

    fn string(&self, out: &mut Vec<u8>, s: &str) {
        self.u32(out, s.len() as u32);
        out.extend_from_slice(s.as_bytes());
    }

    /// `vec(T)` of between `min` and `max` elements emitted by `f`.
    fn vec(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        min: usize,
        max: usize,
        mut f: impl FnMut(&mut Self, &mut Unstructured, &mut Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        let mut items = Vec::new();
        let mut n = 0;
        while n < min || (n < max && more(u, 4)?) {
            f(self, u, &mut items)?;
            n += 1;
        }
        self.u32(out, n as u32);
        out.extend(items);
        Ok(())
    }

    /// `<T>?` where `f` emits `T`, which is absent when `present` is false.
    fn optional(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        present: bool,
        f: impl FnOnce(&mut Self, &mut Unstructured, &mut Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        if present && u.arbitrary()? {
            out.push(0x01);
            f(self, u, out)
        } else {
            out.push(0x00);
            Ok(())
        }
    }

    /// `section_N(B)` or `namesubsection_N(B)` where `f` emits `B`.
    fn subsection(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        id: u8,
        f: impl FnOnce(&mut Self, &mut Unstructured, &mut Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        let mut body = Vec::new();
        f(self, u, &mut body)?;
        out.push(id);
        self.u32(out, body.len() as u32);
        out.extend(body);
        Ok(())
    }

    fn component(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        out.extend([0x00, 0x61, 0x73, 0x6d, 0x0d, 0x00, 0x01, 0x00]);
        self.scopes.push(Scope::default());
        while more(u, 16)? {
            self.section(u, out)?;
        }
        self.scopes.pop();
        Ok(())
    }

    fn section(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let mut ids = vec![2, 3, 5, 7, 8, 10];
        if self.config.generate_custom_sections {
            ids.push(0);
        }
        if self.modules < self.config.max_modules {
            ids.push(1);
        }
        if self.components < self.config.max_components && self.can_nest() {
            ids.push(4);
        }
        if self.can_alias() {
            ids.push(6);
        }
        if self.has(Sort::Func) && !self.scopes.last().unwrap().start {
            ids.push(9);
        }
        if self.has_any(EXTERN_SORTS) {
            ids.push(11);
        }
        let id = *u.choose(&ids)?;
        let c = &self.config;
        let (types, imports, exports, aliases) =
            (c.max_types, c.max_imports, c.max_exports, c.max_aliases);
        let (instances, funcs) = (c.max_instances, c.max_funcs);
        self.subsection(u, out, id, |me, u, out| match id {
            0 => me.custom(u, out),
            1 => me.core_module(u, out),
            2 => me.vec(u, out, 0, instances, Self::core_instance),
            3 => me.vec(u, out, 0, types, Self::core_type),
            4 => {
                me.components += 1;
                me.component(u, out)?;
                me.add(Sort::Component, 1);
                Ok(())
            }
            5 => me.vec(u, out, 0, instances, Self::instance),
            6 => me.vec(u, out, 0, aliases, Self::alias),
            7 => me.vec(u, out, 0, types, Self::ty),
            8 => me.vec(u, out, 0, funcs, Self::canon),
            9 => me.start(u, out),
            10 => me.vec(u, out, 0, imports, Self::import),
            _ => me.vec(u, out, 0, exports, Self::export),
        })
    }

    fn custom(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let name = limited_str(20, u)?;
        // Arbitrary contents are only well-formed in other custom sections.
        if name == "component-name" || u.arbitrary()? {
            return self.namedata(u, out);
        }
        self.string(out, name);
        let len = u.arbitrary_len::<u8>()?.min(100);
        out.extend_from_slice(u.bytes(len)?);
        Ok(())
    }

    fn namedata(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        self.string(out, "component-name");
        if u.arbitrary()? {
            self.subsection(u, out, 0, |me, u, out| {
                let name = limited_str(20, u)?;
                me.string(out, name);
                Ok(())
            })?;
        }
        for &sort in ALL_SORTS {
            if !u.arbitrary()? {
                continue;
            }
            self.subsection(u, out, 1, |me, u, out| {
                sort.encode(out);
                me.namemap(u, out, sort)
            })?;
        }
        Ok(())
    }

    /// `namemap`, whose indices are in increasing order.
    fn namemap(&mut self, u: &mut Unstructured, out: &mut Vec<u8>, sort: Sort) -> Result<()> {
        let mut names = Vec::new();
        let mut n = 0;
        for idx in 0..self.len(sort) {
            if !more(u, 8)? {
                break;
            }
            if u.arbitrary()? {
                self.u32(&mut names, idx);
                let name = limited_str(20, u)?;
                self.string(&mut names, name);
                n += 1;
            }
        }
        self.u32(out, n);
        out.extend(names);
        Ok(())
    }

    fn core_module(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        self.modules += 1;
        let module =
            Module::new_internal(self.config.clone(), u, DuplicateImportsBehavior::Disallowed)?;
        self.core_names
            .extend(module.import_export_names().into_iter().map(String::from));
        out.extend(module.to_bytes());
        self.add(Sort::CoreModule, 1);
        Ok(())
    }

    /// A `core:name` that is usually one a core module uses, and that is made
    /// unique in `used` when given.
    fn core_name(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        used: Option<&mut HashSet<String>>,
    ) -> Result<()> {
        let mut name = if !self.core_names.is_empty() && u.ratio(3, 4)? {
            u.choose(&self.core_names)?.clone()
        } else {
            limited_str(20, u)?.to_string()
        };
        if let Some(used) = used {
            let mut i = used.len();
            while !used.insert(name.clone()) {
                name = format!("x{i}");
                i += 1;
            }
        }
        self.string(out, &name);
        Ok(())
    }

    fn core_instance(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let mut used = HashSet::new();
        if self.has(Sort::CoreModule) && u.arbitrary()? {
            out.push(0x00);
            self.index(u, out, Sort::CoreModule)?;
            let max = if self.has(Sort::CoreInstance) {
                usize::MAX
            } else {
                0
            };
            self.vec(u, out, 0, max, |me, u, out| {
                me.core_name(u, out, Some(&mut used))?;
                out.push(0x12);
                me.index(u, out, Sort::CoreInstance)
            })?;
        } else {
            out.push(0x01);
            let max = if self.has_any(CORE_EXTERN_SORTS) {
                usize::MAX
            } else {
                0
            };
            self.vec(u, out, 0, max, |me, u, out| {
                me.core_name(u, out, Some(&mut used))?;
                let sort = me.pick_sort(u, CORE_EXTERN_SORTS)?;
                out.push(sort.core().unwrap());
                me.index(u, out, sort)
            })?;
        }
        self.add(Sort::CoreInstance, 1);
        Ok(())
    }

    fn core_type(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let mut choices = vec![0];
        if self.config.gc_enabled {
            choices.push(1);
        }
        if self.can_nest() {
            choices.push(2);
        }
        match *u.choose(&choices)? {
            0 => self.core_rectype(u, out, false),
            1 => {
                out.extend([0x00, 0x50]);
                self.core_supertypes(u, out)?;
                self.core_comptype(u, out, 1)?;
                self.add(Sort::CoreType, 1);
                Ok(())
            }
            _ => self.core_moduletype(u, out),
        }
    }

    /// `core:rectype`, where a lone `0x50` subtype is only allowed with
    /// `open_ok` since it is ambiguous with `core:moduletype` otherwise.
    fn core_rectype(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        open_ok: bool,
    ) -> Result<()> {
        // `wast` doesn't parse the `(core rec ...)` that `wasmprinter` prints
        // for these in component and instance types.
        let decl = self.scopes.last().unwrap().decl;
        if self.config.gc_enabled && !decl && u.arbitrary()? {
            out.push(0x4e);
            let n = u.int_in_range(0..=4)?;
            self.u32(out, n);
            for _ in 0..n {
                self.core_subtype(u, out, n, true)?;
            }
            self.add(Sort::CoreType, n);
        } else {
            self.core_subtype(u, out, 1, open_ok)?;
            self.add(Sort::CoreType, 1);
        }
        Ok(())
    }

    /// `core:subtype` in a recursion group of `group` types.
    fn core_subtype(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        group: u32,
        open_ok: bool,
    ) -> Result<()> {
        if self.config.gc_enabled {
            match u.int_in_range(0..=2)? {
                0 if open_ok => {
                    out.push(0x50);
                    self.core_supertypes(u, out)?;
                }
                1 => {
                    out.push(0x4f);
                    self.core_supertypes(u, out)?;
                }
                _ => {}
            }
        }
        self.core_comptype(u, out, group)
    }

    fn core_supertypes(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let max = if self.has(Sort::CoreType) { 1 } else { 0 };
        self.vec(u, out, 0, max, |me, u, out| {
            me.index(u, out, Sort::CoreType)
        })
    }

    fn core_comptype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>, group: u32) -> Result<()> {
        let kind = if self.config.gc_enabled {
            u.int_in_range(0..=2)?
        } else {
            0
        };
        match kind {
            0 => {
                out.push(0x60);
                for _ in 0..2 {
                    self.vec(u, out, 0, usize::MAX, |me, u, out| {
                        me.core_valtype(u, group)?.encode(out);
                        Ok(())
                    })?;
                }
                Ok(())
            }
            1 => {
                out.push(0x5f);
                self.vec(u, out, 0, usize::MAX, |me, u, out| {
                    me.core_fieldtype(u, out, group)
                })
            }
            _ => {
                out.push(0x5e);
                self.core_fieldtype(u, out, group)
            }
        }
    }

    fn core_fieldtype(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        group: u32,
    ) -> Result<()> {
        match u.int_in_range(0..=3)? {
            0 => out.push(0x78),
            1 => out.push(0x77),
            _ => self.core_valtype(u, group)?.encode(out),
        }
        out.push(u.int_in_range(0..=1)?);
        Ok(())
    }

    /// A `core:valtype`, which may refer to any of the `group` types of the
    /// recursion group being defined.
    fn core_valtype(&mut self, u: &mut Unstructured, group: u32) -> Result<ValType> {
        let types = self.len(Sort::CoreType).saturating_add(group);
        if self.config.gc_enabled && types > 0 && u.ratio(1, 4)? {
            return Ok(ValType::Ref(RefType {
                nullable: u.arbitrary()?,
                heap_type: HeapType::Concrete(u.int_in_range(0..=types - 1)?),
            }));
        }
        Ok(*u.choose(&self.core_valtypes)?)
    }

    /// A `core:valtype` that validation requires to be `i32` where it is used.
    fn core_valtype_i32(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let ty = if u.ratio(1, 8)? {
            self.core_valtype(u, 0)?
        } else {
            ValType::I32
        };
        ty.encode(out);
        Ok(())
    }

    fn core_moduletype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        out.push(0x50);
        self.scopes.push(Scope::default());
        let max = self.config.max_types;
        self.vec(u, out, 0, max, Self::core_moduledecl)?;
        self.scopes.pop();
        self.add(Sort::CoreType, 1);
        Ok(())
    }

    fn core_moduledecl(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let mut choices = vec![0, 1, 3];
        if !self.outer_targets(&[Sort::CoreType]).is_empty() {
            choices.push(2);
        }
        match *u.choose(&choices)? {
            0 => {
                out.push(0x00);
                // Unique field names keep the module and field name pairs
                // unique, as validation requires.
                self.core_name(u, out, None)?;
                let mut used = mem::take(&mut self.scope().imports);
                let name = self.core_name(u, out, Some(&mut used));
                self.scope().imports = used;
                name?;
                self.core_externtype(u, out)
            }
            1 => {
                out.push(0x01);
                self.core_rectype(u, out, true)
            }
            2 => {
                out.extend([0x02, 0x10, 0x01]);
                self.outer(u, out, &[Sort::CoreType])?;
                self.add(Sort::CoreType, 1);
                Ok(())
            }
            _ => {
                out.push(0x03);
                let mut used = mem::take(&mut self.scope().exports);
                let name = self.core_name(u, out, Some(&mut used));
                self.scope().exports = used;
                name?;
                self.core_externtype(u, out)
            }
        }
    }

    fn core_externtype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let mut choices = vec![3];
        if self.config.max_tables > 0 {
            choices.push(1);
        }
        if self.config.max_memories > 0 {
            choices.push(2);
        }
        if self.has(Sort::CoreType) {
            choices.push(0);
            if self.config.exceptions_enabled {
                choices.push(4);
            }
        }
        match *u.choose(&choices)? {
            0 => {
                out.push(0x00);
                self.index(u, out, Sort::CoreType)
            }
            1 => {
                out.push(0x01);
                arbitrary_table_type(u, &self.config, None)?.encode(out);
                Ok(())
            }
            2 => {
                out.push(0x02);
                arbitrary_memtype(u, &self.config)?.encode(out);
                Ok(())
            }
            3 => {
                out.push(0x03);
                GlobalType {
                    val_type: self.core_valtype(u, 0)?,
                    mutable: u.arbitrary()?,
                    shared: false,
                }
                .encode(out);
                Ok(())
            }
            _ => {
                out.extend([0x04, 0x00]);
                self.index(u, out, Sort::CoreType)
            }
        }
    }

    /// The enclosing scopes, as outer alias `ct`s, that have an index space of
    /// one of `sorts` that is not empty.
    fn outer_targets(&self, sorts: &[Sort]) -> Vec<(usize, Sort)> {
        let mut targets = Vec::new();
        for (ct, scope) in self.scopes.iter().rev().enumerate() {
            for &sort in sorts {
                if scope.lens[sort as usize] > 0 {
                    targets.push((ct, sort));
                }
            }
        }
        targets
    }

    /// The `ct` and `idx` of an outer alias of one of `sorts`.
    fn outer(&mut self, u: &mut Unstructured, out: &mut Vec<u8>, sorts: &[Sort]) -> Result<Sort> {
        let targets = self.outer_targets(sorts);
        let (ct, sort) = if targets.is_empty() {
            (0, *u.choose(sorts)?)
        } else {
            *u.choose(&targets)?
        };
        self.u32(out, ct as u32);
        let idx = self.idx(u, sort, ct)?;
        self.u32(out, idx);
        Ok(sort)
    }

    fn instance(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let max = if self.has_any(EXTERN_SORTS) {
            usize::MAX
        } else {
            0
        };
        let mut used = HashSet::new();
        if self.has(Sort::Component) && u.arbitrary()? {
            out.push(0x00);
            self.index(u, out, Sort::Component)?;
            self.vec(u, out, 0, max, |me, u, out| {
                let name = me.pooled_name(u, Names::All)?;
                let name = me.unique(name, &mut used);
                me.string(out, &name);
                me.sortidx(u, out)?;
                Ok(())
            })?;
        } else {
            out.push(0x01);
            self.vec(u, out, 0, max, |me, u, out| {
                let mut idx = Vec::new();
                let sort = me.sortidx(u, &mut idx)?;
                me.nameattributes(u, out, &mut used, sort)?;
                out.extend(idx);
                Ok(())
            })?;
        }
        self.add(Sort::Instance, 1);
        Ok(())
    }

    fn sortidx(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<Sort> {
        let sort = self.pick_sort(u, EXTERN_SORTS)?;
        sort.encode(out);
        self.index(u, out, sort)?;
        Ok(sort)
    }

    fn alias_sorts(&self) -> (&'static [Sort], &'static [Sort]) {
        if self.scopes.last().unwrap().decl {
            (DECL_EXPORT_ALIAS_SORTS, DECL_OUTER_ALIAS_SORTS)
        } else {
            (EXTERN_SORTS, OUTER_ALIAS_SORTS)
        }
    }

    fn can_alias(&self) -> bool {
        let (_, outer) = self.alias_sorts();
        self.has(Sort::Instance)
            || self.has(Sort::CoreInstance)
            || !self.outer_targets(outer).is_empty()
    }

    fn alias(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let (export_sorts, outer_sorts) = self.alias_sorts();
        let decl = self.scopes.last().unwrap().decl;
        let mut kinds = Vec::new();
        if self.has(Sort::Instance) {
            kinds.push(0x00);
        }
        if self.has(Sort::CoreInstance) && !decl {
            kinds.push(0x01);
        }
        if kinds.is_empty() || !self.outer_targets(outer_sorts).is_empty() {
            kinds.push(0x02);
        }
        let kind = *u.choose(&kinds)?;
        let mut body = Vec::new();
        let sort = match kind {
            0x00 => {
                self.index(u, &mut body, Sort::Instance)?;
                let name = self.pooled_name(u, Names::All)?;
                self.string(&mut body, &name);
                *u.choose(export_sorts)?
            }
            0x01 => {
                self.index(u, &mut body, Sort::CoreInstance)?;
                self.core_name(u, &mut body, None)?;
                *u.choose(CORE_EXTERN_SORTS)?
            }
            _ => self.outer(u, &mut body, outer_sorts)?,
        };
        sort.encode(out);
        out.push(kind);
        out.extend(body);
        self.add(sort, 1);
        Ok(())
    }

    fn ty(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        self.deftype(u, out)?;
        self.add(Sort::Type, 1);
        Ok(())
    }

    fn deftype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let mut choices = vec![0, 1];
        // Resources can only be defined in components.
        if !self.scopes.last().unwrap().decl {
            choices.push(2);
        }
        if self.can_nest() {
            choices.extend([3, 4]);
        }
        match *u.choose(&choices)? {
            0 => self.defvaltype(u, out),
            1 => self.functype(u, out),
            2 => self.resourcetype(u, out),
            3 => self.decls(u, out, 0x41, Self::componentdecl),
            _ => self.decls(u, out, 0x42, Self::instancedecl),
        }
    }

    fn primvaltype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        out.push(*u.choose(&[
            0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x7a, 0x79, 0x78, 0x77, 0x76, 0x75, 0x74, 0x73, 0x64,
        ])?);
        Ok(())
    }

    fn defvaltype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        // `None` stands for any `primvaltype`.
        let mut ops = vec![
            None,
            Some(0x72),
            Some(0x71),
            Some(0x70),
            Some(0x67),
            Some(0x6f),
            Some(0x6e),
            Some(0x6d),
            Some(0x6b),
            Some(0x6a),
            Some(0x66),
            Some(0x65),
            Some(0x63),
        ];
        if self.has(Sort::Type) {
            ops.extend([Some(0x69), Some(0x68)]);
        }
        let Some(op) = *u.choose(&ops)? else {
            return self.primvaltype(u, out);
        };
        out.push(op);
        let mut labels = HashSet::new();
        match op {
            0x72 => self.vec(u, out, 1, usize::MAX, |me, u, out| {
                me.labellit(u, out, &mut labels)?;
                me.valtype(u, out)
            }),
            0x71 => self.vec(u, out, 1, usize::MAX, |me, u, out| {
                me.labellit(u, out, &mut labels)?;
                me.optional(u, out, true, Self::valtype)?;
                out.push(0x00);
                Ok(())
            }),
            0x67 => {
                self.valtype(u, out)?;
                let len = u.int_in_range(1..=16)?;
                self.u32(out, len);
                Ok(())
            }
            0x6f => self.vec(u, out, 1, usize::MAX, Self::valtype),
            0x6e => self.vec(u, out, 1, 32, |me, u, out| me.labellit(u, out, &mut labels)),
            0x6d => self.vec(u, out, 1, usize::MAX, |me, u, out| {
                me.labellit(u, out, &mut labels)
            }),
            0x6a => {
                self.optional(u, out, true, Self::valtype)?;
                self.optional(u, out, true, Self::valtype)
            }
            0x66 | 0x65 => self.optional(u, out, true, Self::valtype),
            0x63 => {
                // `keytype`
                out.push(*u.choose(&[
                    0x7f, 0x7e, 0x7d, 0x7c, 0x7b, 0x7a, 0x79, 0x78, 0x77, 0x74, 0x73,
                ])?);
                self.valtype(u, out)
            }
            0x69 | 0x68 => self.index(u, out, Sort::Type),
            // `list` and `option`
            _ => self.valtype(u, out),
        }
    }

    /// `valtype`, whose `typeidx` is a non-negative `s33` so that it doesn't
    /// overlap the type opcodes.
    fn valtype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        if self.has(Sort::Type) && u.arbitrary()? {
            i64::from(self.idx(u, Sort::Type, 0)?).encode(out);
            Ok(())
        } else {
            self.primvaltype(u, out)
        }
    }

    fn resourcetype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        out.push(0x3f);
        self.core_valtype_i32(u, out)?;
        let dtor = self.has(Sort::CoreFunc);
        self.optional(u, out, dtor, |me, u, out| me.index(u, out, Sort::CoreFunc))
    }

    fn functype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        out.push(if u.ratio(1, 4)? { 0x43 } else { 0x40 });
        let mut labels = HashSet::new();
        self.vec(u, out, 0, usize::MAX, |me, u, out| {
            me.labellit(u, out, &mut labels)?;
            me.valtype(u, out)
        })?;
        self.resultlist(u, out)
    }

    fn resultlist(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        if u.arbitrary()? {
            out.push(0x00);
            self.valtype(u, out)
        } else {
            out.extend([0x01, 0x00]);
            Ok(())
        }
    }

    /// `componenttype` or `instancetype` with declarations emitted by `decl`.
    fn decls(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        opcode: u8,
        decl: fn(&mut Self, &mut Unstructured, &mut Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        out.push(opcode);
        self.scopes.push(Scope {
            decl: true,
            ..Scope::default()
        });
        let max = self.config.max_types;
        self.vec(u, out, 0, max, decl)?;
        self.scopes.pop();
        Ok(())
    }

    fn componentdecl(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        if u.arbitrary()? {
            out.push(0x03);
            self.externdecl(u, out, true)
        } else {
            self.instancedecl(u, out)
        }
    }

    fn instancedecl(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let mut choices = vec![0x00, 0x01, 0x04];
        if self.can_alias() {
            choices.push(0x02);
        }
        let choice = *u.choose(&choices)?;
        out.push(choice);
        match choice {
            0x00 => self.core_type(u, out),
            0x01 => self.ty(u, out),
            0x02 => self.alias(u, out),
            _ => self.externdecl(u, out, false),
        }
    }

    /// `import`, `importdecl` or `exportdecl`.
    fn externdecl(&mut self, u: &mut Unstructured, out: &mut Vec<u8>, import: bool) -> Result<()> {
        let mut ty = Vec::new();
        let sort = self.externtype(u, &mut ty)?;
        self.scope_nameattributes(u, out, import, sort)?;
        out.extend(ty);
        self.add(sort, 1);
        Ok(())
    }

    fn externtype(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<Sort> {
        let mut ops = vec![0x03];
        if self.has(Sort::CoreType) {
            ops.push(0x00);
        }
        if self.has(Sort::Type) {
            ops.extend([0x01, 0x04, 0x05]);
        }
        let op = *u.choose(&ops)?;
        out.push(op);
        Ok(match op {
            0x00 => {
                out.push(0x11);
                self.index(u, out, Sort::CoreType)?;
                Sort::CoreModule
            }
            0x03 => {
                if self.has(Sort::Type) && u.arbitrary()? {
                    out.push(0x00);
                    self.index(u, out, Sort::Type)?;
                } else {
                    out.push(0x01);
                }
                Sort::Type
            }
            _ => {
                self.index(u, out, Sort::Type)?;
                match op {
                    0x01 => Sort::Func,
                    0x04 => Sort::Component,
                    _ => Sort::Instance,
                }
            }
        })
    }

    fn canon(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        const NONE: &[u8] = &[0x24, 0x25, 0x05, 0x0d, 0x1e, 0x1f, 0x22, 0x23, 0x26, 0x28];
        const ZERO: &[u8] = &[0x29, 0x0c, 0x2a, 0x2b, 0x2c, 0x2d];
        const TYPE: &[u8] = &[
            0x02, 0x03, 0x04, 0x0e, 0x13, 0x14, 0x2e, 0x15, 0x1a, 0x1b, 0x2f,
        ];
        const TYPE_OPTS: &[u8] = &[0x0f, 0x10, 0x16, 0x17];
        const TYPE_ASYNC: &[u8] = &[0x11, 0x12, 0x18, 0x19];
        const OPTS: &[u8] = &[0x1c, 0x1d];

        // Lifting and lowering are the bulk of real components, so pick them
        // half of the time.
        let mut lift_lower = Vec::new();
        if self.has(Sort::CoreFunc) && self.has(Sort::Type) {
            lift_lower.push(0x00);
        }
        if self.has(Sort::Func) {
            lift_lower.push(0x01);
        }
        let op = if !lift_lower.is_empty() && u.arbitrary()? {
            *u.choose(&lift_lower)?
        } else {
            let mut ops = lift_lower;
            ops.extend(NONE.iter().chain(ZERO).chain(OPTS));
            ops.extend([0x09, 0x0a, 0x0b, 0x06]);
            if self.has(Sort::Type) {
                ops.extend(TYPE.iter().chain(TYPE_OPTS).chain(TYPE_ASYNC));
            }
            if self.has(Sort::CoreMemory) {
                ops.extend([0x20, 0x21]);
            }
            if self.has(Sort::CoreType) && self.has(Sort::CoreTable) {
                ops.push(0x27);
            }
            *u.choose(&ops)?
        };

        out.push(op);
        match op {
            0x00 => {
                out.push(0x00);
                self.index(u, out, Sort::CoreFunc)?;
                self.opts(u, out, Opts::Lift)?;
                self.index(u, out, Sort::Type)?;
            }
            0x01 => {
                out.push(0x00);
                self.index(u, out, Sort::Func)?;
                self.opts(u, out, Opts::Lower)?;
            }
            0x09 => {
                self.resultlist(u, out)?;
                self.opts(u, out, Opts::Sync)?;
            }
            0x0a | 0x0b => {
                self.core_valtype_i32(u, out)?;
                let slot = u.int_in_range(0..=1)?;
                self.u32(out, slot);
            }
            0x06 => out.push(u.int_in_range(0..=1)?),
            0x20 | 0x21 => {
                out.push(0x00);
                self.index(u, out, Sort::CoreMemory)?;
            }
            0x27 => {
                self.index(u, out, Sort::CoreType)?;
                self.index(u, out, Sort::CoreTable)?;
            }
            op if ZERO.contains(&op) => out.push(0x00),
            op if TYPE.contains(&op) => self.index(u, out, Sort::Type)?,
            op if TYPE_OPTS.contains(&op) => {
                self.index(u, out, Sort::Type)?;
                self.opts(u, out, Opts::Lower)?;
            }
            op if TYPE_ASYNC.contains(&op) => {
                self.index(u, out, Sort::Type)?;
                out.push(u.int_in_range(0..=1)?);
            }
            op if OPTS.contains(&op) => self.opts(u, out, Opts::Sync)?,
            _ => {}
        }
        self.add(
            if op == 0x00 {
                Sort::Func
            } else {
                Sort::CoreFunc
            },
            1,
        );
        Ok(())
    }

    /// `opts`, with the combinations of `canonopt`s that validation allows.
    fn opts(&mut self, u: &mut Unstructured, out: &mut Vec<u8>, allowed: Opts) -> Result<()> {
        // `None` stands for any `string-encoding`.
        let mut kinds = vec![None];
        if self.has(Sort::CoreMemory) {
            kinds.push(Some(0x03));
            if allowed != Opts::Sync && self.has(Sort::CoreFunc) {
                kinds.push(Some(0x04));
            }
        }
        if allowed != Opts::Sync {
            kinds.push(Some(0x06));
        }
        if allowed == Opts::Lift && self.has(Sort::CoreFunc) {
            kinds.extend([Some(0x05), Some(0x07)]);
        }
        let mut opts = Vec::new();
        while !kinds.is_empty() && more(u, 2)? {
            opts.push(kinds.swap_remove(u.choose_index(kinds.len())?));
        }
        // `realloc` requires `memory`, and `callback` requires `async`, which
        // rejects `post-return`.
        for (opt, requires) in [(0x04, 0x03), (0x07, 0x06)] {
            if opts.contains(&Some(opt)) && !opts.contains(&Some(requires)) {
                opts.push(Some(requires));
            }
        }
        if opts.contains(&Some(0x06)) {
            opts.retain(|opt| *opt != Some(0x05));
        }

        self.u32(out, opts.len() as u32);
        for opt in opts {
            match opt {
                None => out.push(u.int_in_range(0x00..=0x02)?),
                Some(0x03) => {
                    out.push(0x03);
                    self.index(u, out, Sort::CoreMemory)?;
                }
                Some(0x06) => out.push(0x06),
                Some(opt) => {
                    out.push(opt);
                    self.index(u, out, Sort::CoreFunc)?;
                }
            }
        }
        Ok(())
    }

    fn start(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        self.scope().start = true;
        self.index(u, out, Sort::Func)?;
        let max = if self.has(Sort::Value) { usize::MAX } else { 0 };
        self.vec(u, out, 0, max, |me, u, out| me.index(u, out, Sort::Value))?;
        let results = u.int_in_range(0..=1)?;
        self.u32(out, results);
        self.add(Sort::Value, results);
        Ok(())
    }

    fn import(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        self.externdecl(u, out, true)
    }

    fn export(&mut self, u: &mut Unstructured, out: &mut Vec<u8>) -> Result<()> {
        let mut idx = Vec::new();
        let sort = self.sortidx(u, &mut idx)?;
        self.scope_nameattributes(u, out, false, sort)?;
        out.extend(idx);
        let ascribe = u.ratio(1, 8)?;
        self.optional(u, out, ascribe, |me, u, out| {
            me.externtype(u, out)?;
            Ok(())
        })?;
        self.add(sort, 1);
        Ok(())
    }

    /// `nameattributes` of an import or export of `sort` in the current scope.
    fn scope_nameattributes(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        import: bool,
        sort: Sort,
    ) -> Result<()> {
        fn used(scope: &mut Scope, import: bool) -> &mut HashSet<String> {
            if import {
                &mut scope.imports
            } else {
                &mut scope.exports
            }
        }
        let mut names = mem::take(used(self.scope(), import));
        let ret = self.nameattributes(u, out, &mut names, sort);
        *used(self.scope(), import) = names;
        ret
    }

    /// `nameattributes` of `sort` whose `externname` is strongly-unique in
    /// `used`.
    fn nameattributes(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        used: &mut HashSet<String>,
        sort: Sort,
    ) -> Result<()> {
        // 🏷️ `implements` is only valid for instances with a label name, and
        // 🔗 `version-suffix` only for instances, completing the
        // `canonversion` of the name or of `implements` into a `valid semver`.
        let instance = sort == Sort::Instance;
        let implements = instance && u.ratio(1, 4)?;
        let version = if instance && u.ratio(1, 4)? {
            Some(split_semver(u)?)
        } else {
            None
        };
        let name = if implements {
            self.pooled_name(u, Names::Labels)?
        } else if let Some((version, _)) = &version {
            // Not pooled since it is only valid with this suffix.
            self.interfacename(u, Some(version))?
        } else if sort == Sort::Func {
            self.pooled_name(u, Names::All)?
        } else {
            self.pooled_name(u, Names::Unannotated)?
        };
        let name = self.unique(name, used);

        let mut attributes = Vec::new();
        if implements {
            let version = version.as_ref().map(|(version, _)| &version[..]);
            attributes.push((0x00, self.interfacename(u, version)?));
        }
        if let Some((_, suffix)) = version {
            attributes.push((0x01, suffix));
        }
        if u.ratio(1, 8)? {
            attributes.push((0x02, limited_str(20, u)?.to_string()));
        }

        let opcode = if attributes.is_empty() {
            u.int_in_range(0x00..=0x02)?
        } else {
            0x02
        };
        out.push(opcode);
        self.string(out, &name);
        if opcode == 0x02 {
            self.u32(out, attributes.len() as u32);
            while !attributes.is_empty() {
                let (kind, value) = attributes.swap_remove(u.choose_index(attributes.len())?);
                out.push(kind);
                self.string(out, &value);
            }
        }
        Ok(())
    }

    /// An extern name that is usually one generated before.
    fn pooled_name(&mut self, u: &mut Unstructured, names: Names) -> Result<String> {
        let ok = |name: &str| match names {
            Names::All => true,
            Names::Unannotated => !name.starts_with('['),
            Names::Labels => !name.starts_with('[') && !name.contains(':'),
        };
        let pool: Vec<&String> = self.names.iter().filter(|name| ok(name)).collect();
        if !pool.is_empty() && u.ratio(3, 4)? {
            return Ok(u.choose(&pool)?.to_string());
        }
        let name = if names == Names::Labels || u.arbitrary()? {
            plainname(u, names == Names::All)?
        } else {
            self.interfacename(u, None)?
        };
        self.names.push(name.clone());
        Ok(name)
    }

    /// Returns `name`, or a name that is strongly-unique in `used` if `name`
    /// isn't, and adds it to `used`.
    fn unique(&self, mut name: String, used: &mut HashSet<String>) -> String {
        let mut i = used.len();
        while !used.insert(canonical(&name)) {
            name = format!("x{i}");
            i += 1;
        }
        name
    }

    fn labellit(
        &mut self,
        u: &mut Unstructured,
        out: &mut Vec<u8>,
        used: &mut HashSet<String>,
    ) -> Result<()> {
        let label = label(u)?;
        let label = self.unique(label, used);
        self.string(out, &label);
        Ok(())
    }

    /// `interfacename`, whose `interfaceversion` is `version` if given.
    fn interfacename(&mut self, u: &mut Unstructured, version: Option<&str>) -> Result<String> {
        // 🪺 allows more than one namespace and projection.
        let max = if u.ratio(1, 8)? { 3 } else { 1 };
        let mut name = String::new();
        arbitrary_loop(u, 1, max, |u| {
            words(u, &mut name)?;
            name.push(':');
            Ok(true)
        })?;
        words(u, &mut name)?;
        arbitrary_loop(u, 1, max, |u| {
            name.push('/');
            name.push_str(&label(u)?);
            Ok(true)
        })?;
        if let Some(version) = version {
            name.push('@');
            name.push_str(version);
        } else if u.arbitrary()? {
            name.push('@');
            semver(u, &mut name)?;
        }
        Ok(name)
    }
}

/// The canonical form of a name under which names in the same scope must be
/// [strongly-unique].
///
/// [strongly-unique]: https://github.com/WebAssembly/component-model/blob/main/design/mvp/Explainer.md#name-uniqueness
fn canonical(name: &str) -> String {
    let name = name.replace('-', "").to_lowercase();
    let mut rest = &name[..];
    let mut annotations = String::new();
    while let Some(end) = rest.strip_prefix('[').and_then(|r| r.find(']')) {
        let annotation = &rest[..end + 2];
        if annotation == "[constructor]" || annotation == "[set]" {
            annotations.push_str(annotation);
        }
        rest = &rest[end + 2..];
    }
    match rest.split_once('.') {
        Some((a, b)) if a == b => a.to_string(),
        _ => annotations + rest,
    }
}

/// `plainname`, which is annotated only if `annotated`.
fn plainname(u: &mut Unstructured, annotated: bool) -> Result<String> {
    if !annotated || u.arbitrary()? {
        return label(u);
    }
    Ok(match u.int_in_range(0..=4)? {
        0 => format!("[constructor]{}", label(u)?),
        // 📡
        1 => format!("[get]{}", label(u)?),
        2 => format!("[set]{}", label(u)?),
        n => {
            let kind = if n == 3 { "[method]" } else { "[static]" };
            // 📡
            let accessor = *u.choose(&["", "[get]", "[set]"])?;
            format!("{kind}{accessor}{}.{}", label(u)?, label(u)?)
        }
    })
}

/// `label ::= first-fragment ('-' fragment)*`
fn label(u: &mut Unstructured) -> Result<String> {
    let mut label = String::new();
    let acronym = u.arbitrary()?;
    fragment(u, &mut label, acronym, true)?;
    arbitrary_loop(u, 0, 3, |u| {
        label.push('-');
        let acronym = u.arbitrary()?;
        fragment(u, &mut label, acronym, false)?;
        Ok(true)
    })?;
    Ok(label)
}

/// `words ::= first-word ('-' word)*`
fn words(u: &mut Unstructured, out: &mut String) -> Result<()> {
    fragment(u, out, false, true)?;
    arbitrary_loop(u, 0, 2, |u| {
        out.push('-');
        fragment(u, out, false, false)?;
        Ok(true)
    })
}

/// A `word` (`[0-9a-z]+`) or an `acronym` (`[0-9A-Z]+`), which starts with a
/// letter when it is `first`.
fn fragment(u: &mut Unstructured, out: &mut String, acronym: bool, first: bool) -> Result<()> {
    let a = if acronym { b'A' } else { b'a' };
    for i in 0..u.int_in_range(1..=4)? {
        let c = match u.int_in_range(0..=35)? {
            n @ 0..=9 if !(first && i == 0) => b'0' + n,
            n => a + n % 26,
        };
        out.push(char::from(c));
    }
    Ok(())
}

/// `valid semver`
fn semver(u: &mut Unstructured, out: &mut String) -> Result<()> {
    let (major, minor, patch): (u8, u8, u8) = u.arbitrary()?;
    out.push_str(&format!("{major}.{minor}.{patch}"));
    if u.ratio(1, 4)? {
        out.push('-');
        identifiers(u, out)?;
    }
    if u.ratio(1, 4)? {
        out.push('+');
        identifiers(u, out)?;
    }
    Ok(())
}

/// Dot-separated alphanumeric identifiers that don't start with a digit, so
/// that they never have leading zeros.
fn identifiers(u: &mut Unstructured, out: &mut String) -> Result<()> {
    let acronym = u.arbitrary()?;
    fragment(u, out, acronym, true)?;
    arbitrary_loop(u, 0, 2, |u| {
        out.push('.');
        let acronym = u.arbitrary()?;
        fragment(u, out, acronym, true)?;
        Ok(true)
    })
}

/// A 🔗 `canonversion` and the `semversuffix` that completes it into a
/// `valid semver` that canonicalizes into exactly these two parts.
fn split_semver(u: &mut Unstructured) -> Result<(String, String)> {
    let (major, minor, patch): (u8, u8, u8) = u.arbitrary()?;
    let mut build = String::new();
    if u.arbitrary()? {
        build.push('+');
        identifiers(u, &mut build)?;
    }
    let (version, mut suffix) = if u.ratio(1, 4)? {
        let mut version = format!("{major}.{minor}.{patch}-");
        identifiers(u, &mut version)?;
        (version, build)
    } else if major > 0 {
        (major.to_string(), format!(".{minor}.{patch}{build}"))
    } else if minor > 0 {
        (format!("0.{minor}"), format!(".{patch}{build}"))
    } else {
        (format!("0.0.{patch}"), build)
    };
    if suffix.is_empty() {
        suffix.push('+');
        identifiers(u, &mut suffix)?;
    }
    Ok((version, suffix))
}
