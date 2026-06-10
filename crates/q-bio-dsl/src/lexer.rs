//! Lexer for BioDSL
//!
//! Tokenizes BioDSL source code into a stream of tokens.

use logos::Logos;
use std::fmt;

/// BioDSL tokens
#[derive(Logos, Debug, Clone, PartialEq)]
#[logos(skip r"[ \t\r\n\f]+")]
#[logos(skip r"//[^\n]*")]
#[logos(skip r"/\*[^*]*\*+(?:[^/*][^*]*\*+)*/")]
pub enum Token {
    // Keywords
    #[token("molecule")]
    Molecule,

    #[token("genetic_circuit")]
    GeneticCircuit,

    #[token("protein")]
    Protein,

    #[token("scaffold")]
    Scaffold,

    #[token("ring")]
    Ring,

    #[token("substituents")]
    Substituents,

    #[token("stereochemistry")]
    Stereochemistry,

    #[token("synthesis_method")]
    SynthesisMethod,

    #[token("verification")]
    Verification,

    #[token("synthesize")]
    Synthesize,

    #[token("smiles")]
    Smiles,

    #[token("amount")]
    Amount,

    #[token("purity")]
    Purity,

    #[token("quantity")]
    QuantityKw,

    // Genetic circuit keywords
    #[token("promoter")]
    Promoter,

    #[token("gene")]
    Gene,

    #[token("terminator")]
    Terminator,

    #[token("ribosome_binding_site")]
    RibosomeBindingSite,

    #[token("rbs")]
    RBS,

    #[token("input")]
    Input,

    #[token("output")]
    Output,

    #[token("represses")]
    Represses,

    #[token("activates")]
    Activates,

    #[token("fused_to")]
    FusedTo,

    #[token("binding_site")]
    BindingSite,

    #[token("strength")]
    Strength,

    #[token("product")]
    Product,

    #[token("degrades_in")]
    DegradesIn,

    #[token("switches_off")]
    SwitchesOff,

    #[token("switches_on")]
    SwitchesOn,

    // Protein keywords
    #[token("sequence")]
    Sequence,

    #[token("structure")]
    Structure,

    #[token("domain")]
    Domain,

    #[token("helix")]
    Helix,

    #[token("sheet")]
    Sheet,

    #[token("loop")]
    Loop,

    #[token("active_site")]
    ActiveSite,

    #[token("catalytic_triad")]
    CatalyticTriad,

    #[token("binding_pocket")]
    BindingPocket,

    #[token("optimize")]
    Optimize,

    #[token("stability")]
    Stability,

    #[token("solubility")]
    Solubility,

    #[token("expression_host")]
    ExpressionHost,

    #[token("fold_using")]
    FoldUsing,

    // Safety keywords
    #[token("safety")]
    Safety,

    #[token("auxotrophy")]
    Auxotrophy,

    #[token("kill_switch")]
    KillSwitch,

    #[token("generation_limit")]
    GenerationLimit,

    #[token("temperature_sensitive")]
    TemperatureSensitive,

    // Chemistry keywords
    #[token("benzene")]
    Benzene,

    #[token("pyran")]
    Pyran,

    #[token("cyclohexene")]
    Cyclohexene,

    #[token("hydroxyl")]
    Hydroxyl,

    #[token("methyl")]
    Methyl,

    #[token("ethyl")]
    Ethyl,

    #[token("pentyl")]
    Pentyl,

    #[token("amino")]
    Amino,

    #[token("carboxyl")]
    Carboxyl,

    #[token("hydrophobic")]
    Hydrophobic,

    #[token("hydrophilic")]
    Hydrophilic,

    // Stereochemistry (higher priority than Identifier)
    #[token("R", priority = 3)]
    ConfigR,

    #[token("S", priority = 3)]
    ConfigS,

    #[token("E", priority = 3)]
    ConfigE,

    #[token("Z", priority = 3)]
    ConfigZ,

    #[token("chiral_center")]
    ChiralCenter,

    // Robot commands
    #[token("robot_swarm")]
    RobotSwarm,

    #[token("NanoQuantumonas")]
    NanoQuantumonas,

    #[token("atomic_assembly")]
    AtomicAssembly,

    #[token("quantum_tomography")]
    QuantumTomography,

    // Units
    #[token("mg")]
    Milligrams,

    #[token("ug")]
    Micrograms,

    #[token("ng")]
    Nanograms,

    #[token("g", priority = 3)]
    Grams,

    #[token("mol")]
    Moles,

    #[token("mmol")]
    Millimoles,

    #[token("umol")]
    Micromoles,

    #[token("minutes")]
    Minutes,

    #[token("hours")]
    Hours,

    #[token("seconds")]
    Seconds,

    // Boolean
    #[token("true")]
    True,

    #[token("false")]
    False,

    #[token("maximize")]
    Maximize,

    #[token("minimize")]
    Minimize,

    // Symbols
    #[token("{")]
    LBrace,

    #[token("}")]
    RBrace,

    #[token("[")]
    LBracket,

    #[token("]")]
    RBracket,

    #[token("(")]
    LParen,

    #[token(")")]
    RParen,

    #[token(":")]
    Colon,

    #[token(";")]
    Semicolon,

    #[token(",")]
    Comma,

    #[token(".")]
    Dot,

    #[token("@")]
    At,

    #[token("=")]
    Equals,

    #[token("=>")]
    Arrow,

    #[token("|")]
    Pipe,

    #[token(">")]
    GreaterThan,

    #[token("<")]
    LessThan,

    #[token(">=")]
    GreaterEqual,

    #[token("<=")]
    LessEqual,

    #[token("+")]
    Plus,

    #[token("-")]
    Minus,

    #[token("*")]
    Star,

    #[token("/")]
    Slash,

    // Literals
    #[regex(r#""[^"]*""#, |lex| lex.slice()[1..lex.slice().len()-1].to_string())]
    StringLiteral(String),

    #[regex(r"[0-9]+\.[0-9]+([eE][+-]?[0-9]+)?", |lex| lex.slice().parse::<f64>().ok())]
    FloatLiteral(f64),

    #[regex(r"[0-9]+", |lex| lex.slice().parse::<i64>().ok())]
    IntLiteral(i64),

    #[regex(r"0x[0-9a-fA-F]+", |lex| i64::from_str_radix(&lex.slice()[2..], 16).ok())]
    HexLiteral(i64),

    // Identifiers
    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*", |lex| lex.slice().to_string())]
    Identifier(String),
}

impl fmt::Display for Token {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Token::Molecule => write!(f, "molecule"),
            Token::GeneticCircuit => write!(f, "genetic_circuit"),
            Token::Protein => write!(f, "protein"),
            Token::Synthesize => write!(f, "synthesize"),
            Token::StringLiteral(s) => write!(f, "\"{}\"", s),
            Token::FloatLiteral(n) => write!(f, "{}", n),
            Token::IntLiteral(n) => write!(f, "{}", n),
            Token::Identifier(s) => write!(f, "{}", s),
            _ => write!(f, "{:?}", self),
        }
    }
}

/// Lexer wrapper with position tracking
pub struct BioDSLLexer<'source> {
    inner: logos::Lexer<'source, Token>,
    line: usize,
    column: usize,
}

impl<'source> BioDSLLexer<'source> {
    pub fn new(source: &'source str) -> Self {
        Self {
            inner: Token::lexer(source),
            line: 1,
            column: 1,
        }
    }

    /// Get current position
    pub fn position(&self) -> (usize, usize) {
        (self.line, self.column)
    }

    /// Tokenize entire source
    pub fn tokenize(source: &'source str) -> Result<Vec<SpannedToken>, LexerError> {
        let mut lexer = Self::new(source);
        let mut tokens = Vec::new();

        while let Some(result) = lexer.next() {
            match result {
                Ok(token) => {
                    let span = lexer.inner.span();
                    tokens.push(SpannedToken {
                        token,
                        span: Span {
                            start: span.start,
                            end: span.end,
                            line: lexer.line,
                            column: lexer.column,
                        },
                    });
                }
                Err(_) => {
                    let span = lexer.inner.span();
                    let slice = &source[span.clone()];
                    return Err(LexerError::UnexpectedCharacter {
                        char: slice.chars().next().unwrap_or('?'),
                        line: lexer.line,
                        column: lexer.column,
                    });
                }
            }
        }

        Ok(tokens)
    }
}

impl<'source> Iterator for BioDSLLexer<'source> {
    type Item = Result<Token, ()>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
}

/// Token with source location
#[derive(Debug, Clone)]
pub struct SpannedToken {
    pub token: Token,
    pub span: Span,
}

/// Source span
#[derive(Debug, Clone, Copy)]
pub struct Span {
    pub start: usize,
    pub end: usize,
    pub line: usize,
    pub column: usize,
}

/// Lexer error
#[derive(Debug, thiserror::Error)]
pub enum LexerError {
    #[error("Unexpected character '{char}' at line {line}, column {column}")]
    UnexpectedCharacter {
        char: char,
        line: usize,
        column: usize,
    },

    #[error("Unterminated string literal at line {line}")]
    UnterminatedString { line: usize },

    #[error("Invalid number format at line {line}")]
    InvalidNumber { line: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lex_molecule() {
        let source = r#"molecule THC { smiles: "CCCCCC" }"#;
        let tokens = BioDSLLexer::tokenize(source).unwrap();

        assert!(matches!(tokens[0].token, Token::Molecule));
        assert!(matches!(tokens[1].token, Token::Identifier(_)));
        assert!(matches!(tokens[2].token, Token::LBrace));
        assert!(matches!(tokens[3].token, Token::Smiles));
        assert!(matches!(tokens[4].token, Token::Colon));
        assert!(matches!(tokens[5].token, Token::StringLiteral(_)));
        assert!(matches!(tokens[6].token, Token::RBrace));
    }

    #[test]
    fn test_lex_numbers() {
        let source = "42 3.14 0xFF 1e-5";
        let tokens = BioDSLLexer::tokenize(source).unwrap();

        assert!(matches!(tokens[0].token, Token::IntLiteral(42)));
        assert!(matches!(tokens[1].token, Token::FloatLiteral(f) if (f - 3.14).abs() < 0.001));
        assert!(matches!(tokens[2].token, Token::HexLiteral(255)));
        assert!(matches!(tokens[3].token, Token::FloatLiteral(f) if (f - 1e-5).abs() < 1e-10));
    }

    #[test]
    fn test_lex_genetic_circuit() {
        let source = r#"
            genetic_circuit ToggleSwitch {
                promoter pTet { strength: 0.8 }
                gene lacI { represses pTet }
            }
        "#;
        let tokens = BioDSLLexer::tokenize(source).unwrap();

        assert!(matches!(tokens[0].token, Token::GeneticCircuit));
    }
}
