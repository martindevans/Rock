//use std::sync::Arc;
use std::rc::Rc;
use std::collections::hash_map::DefaultHasher;
use std::num::Wrapping;
use std::hash::{ Hash, Hasher };
use std::ops::Deref;
use std::fmt;
use std::cell::{ Cell };

use seahash::SeaHasher;

use num::bigint::{ BigUint };
use num::{ ToPrimitive, FromPrimitive, Integer, Zero };

fn calculate_hash(num: &BigUint) -> u32 {
    let mut s = SeaHasher::new();
    num.hash(&mut s);
    return (s.finish() & 0xFFFF) as u32;
}

fn eq_big_u32(big: &BigUint, i: u32) -> bool {
    return big.bits() <= 32
        && &BigUint::from_u32(i).expect("Failed to convert u32 into BigInt") == big;
}

/* macro_rules! recycle_noun(
 ($nn:expr) => (
    Ok(Rc::new(Box::new($nn)))
 );
 ($nn:expr, $x:expr $( , $more:expr )* ) => (
    try_replace_noun_internals($x, $nn)
        .or_else(|nnn| { return recycle_noun!(nnn); })
 )
);*/

fn try_replace_noun_internals(candidate : Rc<Box<Noun>>, nn: Noun) -> Result<Rc<Box<Noun>>, Noun> {
    if let Ok(mut owned_box) = Rc::try_unwrap(candidate) {
        //we own the box, mutate it in place
        *owned_box = nn;
        return Ok(Rc::new(owned_box));
    } else {
        return Err(nn);
    }
}

pub struct ExecutionContext
{
    Atom0: Rc<Box<Noun>>,
    Atom1: Rc<Box<Noun>>,

    BigInt1: BigUint,
    BigInt2: BigUint,
    BigInt3: BigUint,

    pub Q: Cell<u32>,
    pub E: Cell<u32>,
    pub S: Cell<u32>,
    pub N0: Cell<u32>,
    pub N1: Cell<u32>,
    pub N2: Cell<u32>,
    pub N3: Cell<u32>,
    pub N4: Cell<u32>,
    pub N5: Cell<u32>,
    pub N6: Cell<u32>,
    pub N7: Cell<u32>,
    pub N8: Cell<u32>,
    pub N9: Cell<u32>,
    pub N10: Cell<u32>,
}

impl ExecutionContext {
    pub fn new() -> ExecutionContext {
        return ExecutionContext {
            Atom0: Rc::new(Box::new(Noun::atom_from_u64(0))),
            Atom1: Rc::new(Box::new(Noun::atom_from_u64(1))),

            BigInt1: BigUint::from_u32(1).unwrap(),
            BigInt2: BigUint::from_u32(2).unwrap(),
            BigInt3: BigUint::from_u32(3).unwrap(),

            Q: Cell::new(0),
            E: Cell::new(0),
            S: Cell::new(0),
            N0: Cell::new(0),
            N1: Cell::new(0),
            N2: Cell::new(0),
            N3: Cell::new(0),
            N4: Cell::new(0),
            N5: Cell::new(0),
            N6: Cell::new(0),
            N7: Cell::new(0),
            N8: Cell::new(0),
            N9: Cell::new(0),
            N10: Cell::new(0),
        }
    }
}

#[derive(Debug)]
pub enum Noun {
    Cell(u32, Rc<Box<Noun>>, Rc<Box<Noun>>),
    Atom(u32, BigUint)
}

impl PartialEq for Noun {
    fn eq(&self, other: &Noun) -> bool {
        match (self, other) {
            //Both atoms, compare their values (shortcut with hash comparison first)
            (&Noun::Atom(hl, ref vl), &Noun::Atom(hr, ref vr)) => hl == hr && vl == vr,

            //Both cells, recursively compare the components (shortcut with hash comparison first)
            (&Noun::Cell(hl, ref ll, ref lr), &Noun::Cell(hr, ref rl, ref rr)) => hl == hr && ll == rl && lr == rr,

            //Different types, definitely not equal
            (_, _) => false
        }
    }
}

impl fmt::Display for Noun {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            &Noun::Cell(_, ref l, ref r) => write!(f, "[{} {}]", l, r),
            &Noun::Atom(_, ref v)    => write!(f, "{}", v)
        }
    }
}

impl Noun {
    pub fn atom_from_u64(value: u64) -> Noun {
        let big = BigUint::from_u64(value).expect("Failed to convert u64 into BigInt");
        Noun::Atom(calculate_hash(&big), big)
    }

    pub fn atom_from_biguint(value: BigUint) -> Noun {
        Noun::Atom(calculate_hash(&value), value.clone())
    }

    pub fn cell_from_nouns(left: Noun, right: Noun) -> Noun {
        Noun::cell_from_arcs(
            Rc::new(Box::new(left)),
            Rc::new(Box::new(right))
        )
    }

    fn cell_from_arcs(left: Rc<Box<Noun>>, right: Rc<Box<Noun>>) -> Noun {

        //Calculate the hash of the two child items
        let mut d = DefaultHasher::new();
        d.write_u32(left.get_hash());
        d.write_u32(right.get_hash());
        let hash = (d.finish() & 0xFFFF) as u32;

        Noun::Cell(
            hash,
            left,
            right,
        )
    }

    fn get_hash(&self) -> u32 {
        match self {
            &Noun::Cell(h, _, _) => h,
            &Noun::Atom(h, _)    => h
        }
    }
}

impl Noun {

    // ?[a b]      0
    // ?a          1
    fn question_mark(&self, ctx: &ExecutionContext) -> Rc<Box<Noun>> {

        ctx.Q.set(ctx.Q.get() + 1);

        match self {
            &Noun::Cell(_, _, _) => ctx.Atom0.clone(),
            &Noun::Atom(_, _)    => ctx.Atom1.clone()
        }
    }

    // +[a b]       ERROR
    // +a           1 + a
    fn plus(&self, ctx: &ExecutionContext) -> Option<BigUint> {
        match self {
            &Noun::Cell(_, _, _) => None,
            &Noun::Atom(_, ref n) => {
                return Some(n + 1u8)
            }
        }
    }

    // =[a a]       0
    // =[a b]       1
    // =a           ERROR
    fn equals(&self, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {

        ctx.E.set(ctx.E.get() + 1);

        if let &Noun::Cell(_, ref l, ref r) = self {
            Some(if l == r { ctx.Atom0.clone() } else { ctx.Atom1.clone() })
        } else {
            None
        }
    }

    // /[1 a]           a
    // /[2 a b]         a
    // /[3 a b]         b
    // /[(a + a) b]     /[2 /[a b]]
    // /[(a + a + 1) b] /[3 /[a b]]
    // /a               ERROR
    fn slash(&self, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {
        match self {
            &Noun::Cell(_, ref l, ref r) => Noun::slash_cell(&l, &r, &ctx),
            &Noun::Atom(_, _)            => None
        }
    }

    // /[1 a]           a
    // /[2 a b]         a
    // /[3 a b]         b
    // /[(a + a) b]     /[2 /[a b]]
    // /[(a + a + 1) b] /[3 /[a b]]
    fn slash_cell(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext)-> Option<Rc<Box<Noun>>> {
        if let &Noun::Atom(_, ref lv) = l.deref().deref() {
            Noun::slash_big_num(&lv, &r, &ctx)
        } else {
            None
        }
    }

    // Nock slash op using a native unsigned integer as the selector. Fastest way to perform lookup
    fn slash_num(selector: u64, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {
        
        ctx.S.set(ctx.S.get() + 1);

        let mut node = r;

        let msb = 64 - selector.leading_zeros();
        for i in (0 .. (msb - 1)).rev() {
            if let &Noun::Cell(_, ref ll, ref rr) = node.deref().deref() {
                node = if selector & (1 << i) == 0 { ll } else { rr }
            } else {
                return None;
            }
        }

        return Some(node.clone());
    }

    // Nock slash op using big ints, will fall back to native int if selector fits into a u64
    fn slash_big_num(selector: &BigUint, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {
            
        //try to fall back to the faster slash operator using integers
        //Since `slash_big_num` is recursive we're hopefully going to very rapidly fall back into the fast path even for _vast_ selectors
        if let Some(usel) = selector.to_u64() {
            return Noun::slash_num(usel, &r, &ctx);
        }

        ctx.S.set(ctx.S.get() + 1);

        //Early exit if index is invalid
        if selector.is_zero() { return None; }

        // /[1 a]           a
        if selector == &ctx.BigInt1 {
            return Some(r.clone())
        }

        // /[2 [a b]]       a
        // /[3 [a b]]       b
        let is2 = eq_big_u32(&selector, 2);
        let is3 = eq_big_u32(&selector, 3);
        if is2 || is3 {
            return match r.deref().deref() {
                &Noun::Atom(_, _)    => None,
                &Noun::Cell(_, ref l, ref r) => {
                    Some(if is2 { l.clone() } else { r.clone() })
                }
            }
        }

        // /[(a + a) b]     /[2 /[a b]]
        // /[(a + a + 1) b] /[3 /[a b]]
        if selector.is_even() {
            let half = selector.div_floor(&ctx.BigInt2);

            // /[a b]
            return if let Some(ref inner) = Noun::slash_big_num(&half, &r, &ctx) {
                Noun::slash_big_num(&ctx.BigInt2, &inner, &ctx)
            } else {
                None
            }

        } else {
            let half = selector.div_floor(&ctx.BigInt2);
            
            // /[a b]
            return if let Some(ref inner) = Noun::slash_big_num(&half, &r, &ctx) {
                Noun::slash_big_num(&ctx.BigInt3, &inner, &ctx)
            } else {
                None
            }
        }
    }

    // *[a [b c] d]     [*[a b c] *[a d]]
    // *[a 0 b]         /[b a]
    // *[a 1 b]         b
    // *[a 2 b c]       *[*[a b] *[a c]]
    // *[a 3 b]         ?*[a b]
    // *[a 4 b]         +*[a b]
    // *[a 5 b]         =*[a b]
    // *[a 6 b c d]     *[a 2 [0 1] 2 [1 c d] [1 0] 2 [1 2 3] [1 0] 4 4 b]
    // *[a 7 b c]       *[a 2 b 1 c]
    // *[a 8 b c]       *[a 7 [[7 [0 1] b] 0 1] c]
    // *[a 9 b c]       *[a 7 c 2 [0 1] 0 b]
    // *[a 10 [b c] d]  *[a 8 c 7 [0 3] d]
    // *[a 10 b c]      *[a c]
    //*a                ERROR
    pub fn star(&self, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {
        if let &Noun::Cell(_, ref a, ref r) = self {
            Noun::star_cell(&a, &r, &ctx)
        } else {
            None
        }
    }

    // Computes *[l 2 r] without doing the work of constructing a cell
    fn star2(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {

        ctx.N2.set(ctx.N2.get() + 1);

        if let &Noun::Cell(_, ref b, ref c) = r.deref().deref() {

            let maybe_lb = Noun::star_cell(&l, &b, &ctx);
            let maybe_lc = Noun::star_cell(&l, &c, &ctx);

            return match (maybe_lb, maybe_lc) {
                (Some(lb), Some(lc)) => Noun::star_cell(&lb, &lc, &ctx),
                _ => None
            };
        }

        None
    }

    // Computes the Nock 4 rule:
    //   *[a 4 b]         +*[a b]
    //
    // Given to us in the form [l 4 r]
    fn star4(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {

        ctx.N4.set(ctx.N4.get() + 1);
        let a = l;
        let b = r;

        if let Some(sab) = Noun::star_cell(&a, &b, &ctx) {
            if let Some(num) = sab.plus(&ctx) {

                // Try to overwrite `sab` with the return value if we own the box
                return Some(try_replace_noun_internals(sab, Noun::atom_from_biguint(num))
                    .unwrap_or_else(|n| { return Rc::new(Box::new(n)); }));
            }
        }

        return None;
    }

    // Computes the Nock 6 rule:
    //   *[a 6 b c d]      *[a 2 [0 1] 2 [1 c d] [1 0] 2 [1 2 3] [1 0] 4 4 b]
    //
    // Given to us in the form [l 6 r]
    //
    // Quoting the docs:
    // > 6 is a primitive "if." If b evaluates to 0, we produce c; if b evaluates to 1, we produce d; otherwise, we crash.
    fn star6(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {

        ctx.N6.set(ctx.N6.get() + 1);

        let a = l;

        if let &Noun::Cell(_, ref b, ref cd) = r.deref().deref() {
            if let &Noun::Cell(_, ref c, ref d) = cd.deref().deref() {

                //We've matched down to get [a 6 b c d]
                //Now evaluate *[a b] to get the condition
                if let Some(ref eb) = Noun::star_cell(&a, &b, &ctx) {
                    if let &Noun::Atom(_, ref v) = eb.deref().deref() {
                        return match v.to_u8() {

                            //Now return *[a c] or *[a d] depending on the condition
                            Some(0) => Noun::star_cell(&a, &c, &ctx),
                            Some(1) => Noun::star_cell(&a, &d, &ctx),

                            _ => None
                        };
                    }
                }
            }
        }

        return None;
    }

    // Computes the Nock 7 rule:
    //   *[a 7 b c]        *[a 2 b 1 c]
    // AKA
    //   *[a 7 b c]        *[*[a b] c]
    //
    // Given to us in the form [l 7 r]
    fn star7(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {

        ctx.N7.set(ctx.N7.get() + 1);

        let a = l;

        if let &Noun::Cell(_, ref b, ref c) = r.deref().deref() {
            if let Some(ref sab) = Noun::star_cell(&a, &b, &ctx) {
                return Noun::star_cell(&sab, &c, &ctx);
            }
        }

        return None;
    }

    // Computes the Nock 8 rule:
    //   *[a 8 b c]        *[a 7 [[7 [0 1] b] 0 1] c]
    // AKA
    //   *[a 8 b c]        *[[*[a b] a] c]
    //
    // Given to us in the form [l 8 r]
    fn star8(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {

        ctx.N8.set(ctx.N8.get() + 1);

        let a = l;

        if let &Noun::Cell(_, ref b, ref c) = r.deref().deref() {
            if let Some(ref sab) = Noun::star_cell(&a, &b, &ctx) {

                // create [*[a b] a]
                let saba = Rc::new(Box::new(Noun::cell_from_arcs(
                    sab.clone(),
                    a.clone()
                )));

                return Noun::star_cell(&saba, &c, &ctx);
            }
        }

        return None;
    }

    // Computes the Nock 9 rule:
    //   *[a 9 b c]
    // AKA
    //    *[*[a c] *[*[a c] 0 b]]
    //    *[*[a c] /[b *[a c]]]
    //
    // Given to us in the form [l 9 r]
    fn star9(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {

        ctx.N9.set(ctx.N9.get() + 1);

        let a = l;

        if let &Noun::Cell(_, ref b, ref c) = r.deref().deref() {
            if let Some(ref sac) = Noun::star_cell(&a, &c, &ctx) {
                if let Some(ref slash) = Noun::slash_cell(&b, &sac, &ctx) {
                    return Noun::star_cell(&sac, &slash, &ctx);
                }
            }
        }

        return None;
    }

    // Computes the Nock 10 rules:
    //   *[a 10 [b c] d]  *[a 8 c 7 [0 3] d]
    //   *[a 10 b c]      *[a c]
    //
    // Given to us in the form:
    //   *[l 10 r]
    fn star10(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {

        //First up we need to distinguish which one we're running
        //
        // [a [10 [[b c] d]]]
        // [a [10 [b c]]]

        unimplemented!();
    }

    // Compute *[a, [n, b]]
    fn star_triple(a: &Rc<Box<Noun>>, n: &Rc<Box<Noun>>, b: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {
        match n.deref().deref() {
            &Noun::Cell(_, ref nl, ref nr) => {
                // n is a cell, so we're in this case:
                // *[a [b c] d]     [*[a b c] *[a d]]
                // *[a [nl nr] b]   [*[a nl nr] *[a b]]

                if let Some(anlnr) = Noun::star_triple(&a, &nl, &nr, &ctx) {
                    if let Some(ab) = Noun::star_cell(&a, &b, &ctx) {
                        return Some(Rc::new(Box::new(Noun::cell_from_arcs(anlnr, ab))));
                    }
                }
                
                return None;
            },
            &Noun::Atom(_, ref n) => {
                match n.to_u8() {
                    // *[a 0 b]         /[b a]
                    Some(0) => Noun::slash_cell(&b, &a, &ctx),

                    // *[a 1 b]         b
                    Some(1) => Some(b.clone()),

                    // *[a 2 b c]       *[*[a b] *[a c]]
                    Some(2) => Noun::star2(&a, &b, &ctx),

                    // *[a 3 b]         ?*[a b]
                    Some(3) => Noun::star_cell(&a, &b, &ctx)
                        .and_then(|result| Some(result.question_mark(&ctx))),

                    // *[a 4 b]         +*[a b]
                    Some(4) => Noun::star4(&a, &b, &ctx),

                    // *[a 5 b]         =*[a b]
                    Some(5) => Noun::star_cell(&a, &b, &ctx)
                        .and_then(|result| result.equals(&ctx)),

                    // *[a 6 b c d]     *[a 2 [0 1] 2 [1 c d] [1 0] 2 [1 2 3] [1 0] 4 4 b]
                    Some(6) => Noun::star6(&a, &b, &ctx),

                    // *[a 7 b c]       *[a 2 b 1 c]
                    Some(7) => Noun::star7(&a, &b, &ctx),

                    // *[a 8 b c]       *[a 7 [[7 [0 1] b] 0 1] c]
                    Some(8) => Noun::star8(&a, &b, &ctx),

                    // *[a 9 b c]       *[a 7 c 2 [0 1] 0 b]
                    Some(9) => Noun::star9(&a, &b, &ctx),

                    // *[a 10 [b c] d]  *[a 8 c 7 [0 3] d]
                    // *[a 10 b c]      *[a c]
                    Some(10) => Noun::star10(&a, &b, &ctx),

                    None => None,
                    Some(_) => None,
                }
            }
        }
    }

    // Computes *[a, r] without doing the work of constructing a cell
    fn star_cell(l: &Rc<Box<Noun>>, r: &Rc<Box<Noun>>, ctx: &ExecutionContext) -> Option<Rc<Box<Noun>>> {
        if let &Noun::Cell(_, ref n, ref b) = r.deref().deref() {
            Noun::star_triple(&l, &n, &b, &ctx)
        } else {
            None
        }
    }
}

pub fn nock(root: Noun, ctx: &ExecutionContext) -> Option<Box<Noun>> {

    //Calculate the result
    let result = root.star(&ctx);

    //Drop the input, the only thing left alive will be the result
    drop(root);

    //unwrap that from it's ARC (which now must be 1) and return it
    return result.map(|r| { Rc::try_unwrap(r).expect("Result RC should be 1") });
}

#[cfg(test)]
mod basic_tests {

    mod question_mark_tests {

        use noun::{ Noun, ExecutionContext };

        #[test]
        fn question_mark_one_for_atom() {
            // ?1
            assert_eq!(&Noun::atom_from_u64(1), &**Noun::atom_from_u64(1).question_mark(&ExecutionContext::new()));
        }

        #[test]
        fn question_mark_zero_for_cell() {

            // ?[1, 2]

            let c = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(2)
            );

            assert_eq!(&Noun::atom_from_u64(0), &**c.question_mark(&ExecutionContext::new()));
        }
    }

    mod plus_tests {

        use noun::{ Noun, ExecutionContext };
        use num::bigint::{ BigUint };
        use num::FromPrimitive;

        #[test]
        fn plus_none_for_cell() {

            // +[1, 2]

            let c = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(2)
            );

            assert_eq!(None, c.plus(&ExecutionContext::new()));
        }

        #[test]
        fn plus_adds_one_for_atom() {
            // +1
            assert_eq!(Some(BigUint::from_i32(2).expect("Failed to convert i32 into BigInt")), Noun::atom_from_u64(1).plus(&ExecutionContext::new()));
        }
    }

    mod equals_tests {

        use noun::{ Noun, ExecutionContext };
        use std::ops::Deref;
        use num::ToPrimitive;

        #[test]
        fn equals_none_for_atom() {
            // =1
            assert_eq!(None, Noun::atom_from_u64(1).equals(&ExecutionContext::new()));
        }

        #[test]
        fn equals_true_for_equal_shallow_cell() {

            // =[1, 1]

            let c = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(1)
            );

            assert_eq!(Noun::atom_from_u64(0), **c.equals(&ExecutionContext::new()).unwrap());
        }

        #[test]
        fn equals_true_for_equal_deep_cell() {

            // =[[1, 1], [1, 1]]

            let a = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(1)
            );

            let b = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(1)
            );

            let c = Noun::cell_from_nouns(a, b);

            assert_eq!(Noun::atom_from_u64(0), **c.equals(&ExecutionContext::new()).unwrap());
        }

        #[test]
        fn equals_false_for_differing_shallow_cell() {

            // =[1, 2]

            let c = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(2)
            );

            assert_eq!(Noun::atom_from_u64(1), **c.equals(&ExecutionContext::new()).unwrap());
        }

        #[test]
        fn equals_false_for_differing_deep_cell() {

            // =[[1, 2], [1, 1]]

            let a = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(2)
            );

            let b = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(1)
            );

            let c = Noun::cell_from_nouns(a, b);

            assert_eq!(Noun::atom_from_u64(1), **c.equals(&ExecutionContext::new()).unwrap());
        }
    }

    mod slash_tests {

        use noun::{ Noun, ExecutionContext };
        use std::ops::Deref;

        #[test]
        fn slash_none_for_atom() {
            // /1
            assert_eq!(None, Noun::atom_from_u64(1).slash(&ExecutionContext::new()));
        }

        #[test]
        fn slash_root_for_1() {
            // /[1 2]

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::atom_from_u64(2)
            );

            let result = cell.slash(&ExecutionContext::new());

            assert_eq!(&Noun::atom_from_u64(2), result.unwrap().deref().deref());
        }

        #[test]
        fn slash_left_for_2() {
            // /[2 [3 4]]

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(2),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(3),
                    Noun::atom_from_u64(4)
                )
            );

            let result = cell.slash(&ExecutionContext::new());

            assert_eq!(&Noun::atom_from_u64(3), result.unwrap().deref().deref());
        }

        #[test]
        fn slash_right_for_3() {
            // /[3 [2 [4 4]]]

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(3),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(2),
                    Noun::cell_from_nouns(
                        Noun::atom_from_u64(4),
                        Noun::atom_from_u64(4)
                    )
                )
            );

            let result = cell.slash(&ExecutionContext::new());

            assert_eq!(&Noun::cell_from_nouns(
                Noun::atom_from_u64(4),
                Noun::atom_from_u64(4)
            ), result.unwrap().deref().deref());
        }

        #[test]
        fn slash_path_for_4() {
            // /[4 [2 [1 3]]]

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(4),
                Noun::cell_from_nouns(
                    Noun::cell_from_nouns(
                        Noun::atom_from_u64(1),
                        Noun::atom_from_u64(3)
                    ),
                    Noun::atom_from_u64(2)
                )
            );

            let result = cell.slash(&ExecutionContext::new());

            assert_eq!(&Noun::atom_from_u64(1), result.unwrap().deref().deref());
        }

        #[test]
        fn slash_path_for_5() {
            // /[5 [[1 3] [4 2]]]

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(5),
                Noun::cell_from_nouns(
                    Noun::cell_from_nouns(
                        Noun::atom_from_u64(1),
                        Noun::atom_from_u64(3)
                    ),
                    Noun::cell_from_nouns(
                        Noun::atom_from_u64(4),
                        Noun::atom_from_u64(2)
                    ),
                )
            );

            let result = cell.slash(&ExecutionContext::new());

            assert_eq!(&Noun::atom_from_u64(3), result.unwrap().deref().deref());
        }
    }

    mod star_tests {

        use noun::{ Noun, ExecutionContext };
        use std::ops::Deref;
        use num::bigint::{ BigUint };
        use num::FromPrimitive;
        use num::ToPrimitive;
        use std::rc::Rc;
        
        #[test]    
        fn star_none_for_atom() {
            let result = Noun::atom_from_u64(1).star(&ExecutionContext::new());
            assert_eq!(None, result);
        }

        #[test]    
        fn star_zero() {

            //Testing this rule:
            // *[a 0 b]         /[b a]

            //Set up this test scenario
            // *[[1 3] 0 2]     /[2 [1 3]]      1
            let result = Noun::cell_from_nouns(
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(1),
                    Noun::atom_from_u64(3),
                ),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(0),
                    Noun::atom_from_u64(2),
                )
            ).star(&ExecutionContext::new());

            assert_eq!(&Noun::atom_from_u64(1), result.expect("Expecting Some(1)").deref().deref());
        }

        #[test]    
        fn star_one() {

            //Testing this rule:
            // *[a 1 b]         b

            //Set up this test scenario
            // *[7 [1 8]]       8
            let result = Noun::cell_from_nouns(
                Noun::atom_from_u64(7),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(1),
                    Noun::atom_from_u64(8),
                )
            ).star(&ExecutionContext::new());

            assert_eq!(&Noun::atom_from_u64(8), result.expect("Expecting Some(8)").deref().deref());
        }

        #[test]    
        fn star_two() {

            //Testing this rule:
            // *[a 2 b c]       *[*[a b] *[a c]]

            //Set up this test scenario:
            //
            // *[1 [2 [1 1] [1 [1 7]]]]     7
            //
            //Prove this by running this in the Dojo
            //
            // .*(1 [2 [1 1] [1 [1 7]]])

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(1),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(2),
                    Noun::cell_from_nouns(
                        Noun::cell_from_nouns(
                            Noun::atom_from_u64(1),
                            Noun::atom_from_u64(1)
                        ),
                        Noun::cell_from_nouns(
                            Noun::atom_from_u64(1),
                            Noun::cell_from_nouns(
                                Noun::atom_from_u64(1),
                                Noun::atom_from_u64(7)
                            )
                        )
                    )
                )
            );

            assert_eq!(&Noun::atom_from_u64(7), cell.star(&ExecutionContext::new()).expect("Expecting Some(7)").deref().deref());
        }

        #[test]
        fn star_three() {

            //Testing this rule:
            // *[a 3 b]         ?*[a b]

            //Set up this test scenario:
            //
            // *[7 [3 [1 8]]]       1
            //
            //Prove this by running this in the Dojo
            //
            // .*(7 [3 [1 8]])

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(8),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(3),
                    Noun::cell_from_nouns(
                        Noun::atom_from_u64(1),
                        Noun::atom_from_u64(8)
                    )
                )
            );

            assert_eq!(&Noun::atom_from_u64(1), &**cell.star(&ExecutionContext::new()).expect("Expecting Some(1)"));
        }

        #[test]
        fn star_four() {

            //Testing this rule:
            // *[a 4 b]         +*[a b]

            //Set up this test scenario:
            //
            // *[7 [4 [1 8]]]       9
            //
            //Prove this by running this in the Dojo
            //
            // .*(7 [4 [1 8]])

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(7),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(4),
                    Noun::cell_from_nouns(
                        Noun::atom_from_u64(1),
                        Noun::atom_from_u64(8)
                    )
                )
            );

            assert_eq!(&Noun::atom_from_u64(9), &**cell.star(&ExecutionContext::new()).expect("Expecting Some(9)"));
        }

        #[test]
        fn star_five() {

            //Testing this rule:
            // *[a 5 b]         =*[a b]

            //Set up this test scenario:
            //
            // *[7 [5 [1 [8, 8]]]]       0
            //
            //Prove this by running this in the Dojo
            //
            // .*(7 [5 [1 [8 8]])

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(7),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(5),
                    Noun::cell_from_nouns(
                        Noun::atom_from_u64(1),
                        Noun::cell_from_nouns(
                            Noun::atom_from_u64(8),
                            Noun::atom_from_u64(8)
                        )
                    )
                )
            );

            assert_eq!(&Noun::atom_from_u64(0), &**cell.star(&ExecutionContext::new()).expect("Expecting Some(0)"));
        }

        #[test]
        fn star_six_1() {
            //Testing this rule:
            // *[a 6 b c d]      *[a 2 [0 1] 2 [1 c d] [1 0] 2 [1 2 3] [1 0] 4 4 b]

            //Using example from the docs at https://github.com/cgyarvin/urbit/blob/master/doc/book/1-nock.markdown
            //
            // .*(42 [6 [1 0] [4 0 1] [1 233]])             43

            let c10 = Noun::cell_from_nouns(Noun::atom_from_u64(1), Noun::atom_from_u64(0));    // [1 0]

            let c401 = Noun::cell_from_nouns(Noun::atom_from_u64(4), Noun::cell_from_nouns(Noun::atom_from_u64(0), Noun::atom_from_u64(1)));    // [4 0 1]

            let c1233 = Noun::cell_from_nouns(Noun::atom_from_u64(1), Noun::atom_from_u64(233));    // [1 233]

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(42),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(6),
                    Noun::cell_from_nouns(
                        c10,
                        Noun::cell_from_nouns(
                            c401,
                            c1233
                        )
                    )
                )
            );

            assert_eq!(&Noun::atom_from_u64(43), &**cell.star(&ExecutionContext::new()).expect("Expecting Some(43)"));
        }

        #[test]
        fn star_six_2() {
            //Testing this rule:
            // *[a 6 b c d]      *[a 2 [0 1] 2 [1 c d] [1 0] 2 [1 2 3] [1 0] 4 4 b]

            //Using example from the docs at https://github.com/cgyarvin/urbit/blob/master/doc/book/1-nock.markdown
            //
            // .*(42 [6 [1 1] [4 0 1] [1 233]])             233

            let c11 = Noun::cell_from_nouns(Noun::atom_from_u64(1), Noun::atom_from_u64(1));    // [1 1]
            let c401 = Noun::cell_from_nouns(Noun::atom_from_u64(4), Noun::cell_from_nouns(Noun::atom_from_u64(0), Noun::atom_from_u64(1)));    // [4 0 1]
            let c1233 = Noun::cell_from_nouns(Noun::atom_from_u64(1), Noun::atom_from_u64(233));    // [1 233]
            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(42),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(6),
                    Noun::cell_from_nouns(
                        c11,
                        Noun::cell_from_nouns(
                            c401,
                            c1233
                        )
                    )
                )
            );

            assert_eq!(&Noun::atom_from_u64(233), &**cell.star(&ExecutionContext::new()).expect("Expecting Some(43)"));
        }

        #[test]
        fn star_seven() {
            // Testing this rule:
            // *[a 7 b c]        *[a 2 b 1 c]

            //Using this example from the docs at https://github.com/cgyarvin/urbit/blob/master/doc/book/1-nock.markdown
            //
            // .*(42 [7 [4 0 1] [4 0 1]])               44

            let c401 = Rc::new(Box::new(Noun::cell_from_nouns(Noun::atom_from_u64(4), Noun::cell_from_nouns(Noun::atom_from_u64(0), Noun::atom_from_u64(1)))));    // [4 0 1]

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(42),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(7),
                    Noun::cell_from_arcs(
                        c401.clone(),
                        c401.clone()
                    )
                )
            );

            assert_eq!(&Noun::atom_from_u64(44), &**cell.star(&ExecutionContext::new()).expect("Expecting Some(44)"));
        }

        #[test]
        fn star_eight_1() {
            // Testing this rule:
            // *[a 8 b c]        *[a 7 [[7 [0 1] b] 0 1] c]

            //Using this example from the docs at https://github.com/cgyarvin/urbit/blob/master/doc/book/1-nock.markdown
            //
            // .*(42 [8 [4 0 1] [0 1]])             [43 42]

            let c01 = Rc::new(Box::new(Noun::cell_from_nouns(Noun::atom_from_u64(0), Noun::atom_from_u64(1))));
            let c401 = Rc::new(Box::new(Noun::cell_from_nouns(Noun::atom_from_u64(4), Noun::cell_from_nouns(Noun::atom_from_u64(0), Noun::atom_from_u64(1)))));
            

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(42),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(8),
                    Noun::cell_from_arcs(
                        c401,
                        c01
                    )
                )
            );

            assert_eq!(&Noun::cell_from_nouns(Noun::atom_from_u64(43), Noun::atom_from_u64(42)), &**cell.star(&ExecutionContext::new()).expect("Expecting Some(44)"));
        }

        #[test]
        fn star_eight_2() {
            // Testing this rule:
            // *[a 8 b c]        *[a 7 [[7 [0 1] b] 0 1] c]

            //Using this example from the docs at https://github.com/cgyarvin/urbit/blob/master/doc/book/1-nock.markdown
            //
            // .*(42 [8 [4 0 1] [4 0 3]])               43

            let c403 = Rc::new(Box::new(Noun::cell_from_nouns(Noun::atom_from_u64(4), Noun::cell_from_nouns(Noun::atom_from_u64(0), Noun::atom_from_u64(3)))));
            let c401 = Rc::new(Box::new(Noun::cell_from_nouns(Noun::atom_from_u64(4), Noun::cell_from_nouns(Noun::atom_from_u64(0), Noun::atom_from_u64(1)))));
            

            let cell = Noun::cell_from_nouns(
                Noun::atom_from_u64(42),
                Noun::cell_from_nouns(
                    Noun::atom_from_u64(8),
                    Noun::cell_from_arcs(
                        c401,
                        c403
                    )
                )
            );

            assert_eq!(&Noun::atom_from_u64(43), &**cell.star(&ExecutionContext::new()).expect("Expecting Some(44)"));
        }
    }
}