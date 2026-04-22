use syn::visit_mut::VisitMut;
use syn::{parse_quote, Attribute, Block, File, ItemFn, ItemMod, ImplItemFn, ReturnType, Type, Signature};

pub fn stub_file(source: &str) -> Result<String, syn::Error> {
    let mut ast: File = syn::parse_str(source)?;
    let mut visitor = StubVisitor;
    visitor.visit_file_mut(&mut ast);
    Ok(prettyplease::unparse(&ast))
}

struct StubVisitor;

fn has_test_attr(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| attr.path().is_ident("test"))
}

fn has_cfg_test(attrs: &[Attribute]) -> bool {
    attrs.iter().any(|attr| {
        if !attr.path().is_ident("cfg") {
            return false;
        }
        attr.parse_args::<syn::Ident>()
            .map(|ident| ident == "test")
            .unwrap_or(false)
    })
}

/// `todo!()` returns `!` which doesn't satisfy `impl Trait` on stable Rust,
/// so we skip stubbing functions that return `impl Trait`.
fn returns_impl_trait(output: &ReturnType) -> bool {
    match output {
        ReturnType::Default => false,
        ReturnType::Type(_, ty) => type_contains_impl_trait(ty),
    }
}

fn type_contains_impl_trait(ty: &Type) -> bool {
    match ty {
        Type::ImplTrait(_) => true,
        Type::Paren(inner) => type_contains_impl_trait(&inner.elem),
        Type::Group(inner) => type_contains_impl_trait(&inner.elem),
        Type::Reference(inner) => type_contains_impl_trait(&inner.elem),
        _ => false,
    }
}

/// `todo!()` expands to formatting macros which aren't allowed in const context.
/// Use `panic!("STUB")` with a string literal instead (allowed since Rust 1.57).
fn stub_block_for_sig(sig: &Signature) -> Block {
    if sig.constness.is_some() {
        parse_quote!({ panic!("STUB") })
    } else {
        parse_quote!({ todo!("STUB") })
    }
}

fn should_stub_fn(attrs: &[Attribute], output: &ReturnType) -> bool {
    !has_test_attr(attrs) && !returns_impl_trait(output)
}

impl VisitMut for StubVisitor {
    fn visit_item_fn_mut(&mut self, node: &mut ItemFn) {
        if should_stub_fn(&node.attrs, &node.sig.output) {
            *node.block = stub_block_for_sig(&node.sig);
        }
    }

    fn visit_impl_item_fn_mut(&mut self, node: &mut ImplItemFn) {
        if should_stub_fn(&node.attrs, &node.sig.output) {
            node.block = stub_block_for_sig(&node.sig);
        }
    }

    fn visit_item_mod_mut(&mut self, node: &mut ItemMod) {
        if has_cfg_test(&node.attrs) {
            return;
        }
        syn::visit_mut::visit_item_mod_mut(self, node);
    }
}
