use noun::{ Noun, nock };

#[derive(Debug, PartialEq)]
enum Token {
    Left,
    Right,
    Integer(u64)
}

pub fn parse(input: String) -> Noun {

    let mut tokens : Vec<Token>= input
        .replace("[", " [ ")
        .replace("]", " ] ")
        .split_whitespace()
        .map(|c| {
            match c {
                "[" => Token::Left,
                "]" => Token::Right,
                value => Token::Integer(value.parse::<u64>().expect("Not an integer"))
            }
        })
        .collect();

    consume(Token::Left, &mut tokens);
    return parse_cell(&mut tokens);
    
}

fn consume(tok: Token, input: &mut Vec<Token>) {

    if input[0] != tok {
        panic!("Encountered unexpected parsing token. Expected {:?}, got {:?}", tok, input[0]);
    }

    input.remove(0);
}

fn parse_cell(input: &mut Vec<Token>) -> Noun {

    let mut children = Vec::<Noun>::new();

    while input.len() > 0 && input[0] != Token::Right {
        let tok = input.remove(0);

        let n = match tok {
            Token::Integer(value) => Noun::atom_from_u64(value),
            Token::Left => parse_cell(input),
            Token::Right => panic!("Encountered unexpected parsing token. Right bracket not valid here")
        };

        children.push(n);
    }

    consume(Token::Right, input);

    return form_cell(&mut children);
}

fn form_cell(mut nouns: &mut Vec<Noun>) -> Noun {

    if nouns.len() < 2 {
        panic!("Cannot form a cell from less than 2 child nouns");
    }

    if nouns.len() == 2 {
        return Noun::cell_from_nouns(
            nouns.remove(0),
            nouns.remove(0)
        )
    }

    let n0 = nouns.remove(0);
    let n1 = form_cell(&mut nouns);

    return Noun::cell_from_nouns(n0, n1);
}