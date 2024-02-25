"""Test Neurolucida ASC format."""

from io import StringIO

from swcgeom.transforms.neurolucida_asc import (
    AST,
    ASCColor,
    ASCNode,
    ASTNode,
    ASTType,
    Lexer,
    NeurolucidaAscToSwc,
    Parser,
)


class TestNeurolucidaAscToSwc:
    """Test NeurolucidaAscToSwc."""

    def test_simple_tree(self):
        asc = StringIO(
            """
            (
                (Axon)
                (1 0 0 1)
                (2 0 0 1)
                ( 
                    (3 1 0 1)
                    (3 2 0 1)
                    |
                    (3 0 1 1)
                    (3 0 2 1)
                )
            )
            """
        )
        tree = NeurolucidaAscToSwc.from_stream(asc)
        assert tree.number_of_nodes() == 6


class TestAscParser:
    def test_parser_tokens(self):
        """Test parser with different types of tokens."""
        asc = StringIO(
            """
            (
                (Color Red)
                (Dendrite)
                (1 0 0 1)
                (2 0 0 1)
            )
            """
        )
        parser = Parser(asc)
        ast = parser.parse()

        # fmt: off
        assert ast == AST([
            ASTNode(ASTType.COLOR, ASCColor("RED")),
            ASTNode(ASTType.TREE, "DENDRITE", children=[
                ASTNode(ASTType.NODE, ASCNode(1, 0, 0, 1), children=[
                    ASTNode(ASTType.NODE, ASCNode(2, 0, 0, 1))
                ])
            ])
        ])
        # fmt: on

    def test_parser_bif_tree(self):
        """Test parser with bifurcating tree."""
        asc = StringIO(
            """
            (
                (Axon)
                (1 0 0 1)
                (2 0 0 1)
                ( 
                    (3 1 0 1)
                    (3 2 0 1)
                    |
                    (3 0 1 1)
                    (3 0 2 1)
                )
            )
            """
        )
        parser = Parser(asc)
        ast = parser.parse()

        # fmt: off
        assert ast == AST([
            ASTNode(ASTType.TREE, "AXON", children=[
                ASTNode(ASTType.NODE, ASCNode(1, 0, 0, 1), children=[
                    ASTNode(ASTType.NODE, ASCNode(2, 0, 0, 1), children=[
                        ASTNode(ASTType.NODE, ASCNode(3, 1, 0, 1), children=[
                            ASTNode(ASTType.NODE, ASCNode(3, 2, 0, 1)),
                        ]),
                        ASTNode(ASTType.NODE, ASCNode(3, 0, 1, 1), children=[
                            ASTNode(ASTType.NODE, ASCNode(3, 0, 2, 1)),
                        ])
                    ])
                ])
            ])
        ])
        # fmt: on

    def test_parser_empty_branch(self):
        """Test parser with empty branch."""
        asc = StringIO(
            """
            (
                (Axon)
                (1 0 0 1)
                (
                    (2 0 0 1)
                    |
                )
            )
            """
        )
        parser = Parser(asc)
        ast = parser.parse()
        # fmt: off
        assert ast == AST([
            ASTNode(ASTType.TREE, "AXON", children=[
                ASTNode(ASTType.NODE, ASCNode(1, 0, 0, 1), children=[
                    ASTNode(ASTType.NODE, ASCNode(2, 0, 0, 1))
                ])
            ])
        ])
        # fmt: on


class TestAscLexer:
    def test_lexer_floats(self):
        s = StringIO("0 1 -2 3 4.5 -6.7")
        lexer = Lexer(s)
        words = [token.value for token in lexer]
        assert words == [0, 1, -2, 3, 4.5, -6.7]

    def test_lexer_comment(self):
        s = StringIO(
            """
            1   ; comment a
            AXON; comment b
            ;
            """
        )
        lexer = Lexer(s)
        words = [token.value for token in lexer]
        assert words == [1, " comment a", "AXON", " comment b", ""]
