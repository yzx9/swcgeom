"""Neurolucida related transformation."""

import os
import re
from enum import Enum, auto
from io import TextIOBase
from typing import Any, List, NamedTuple, Optional, cast

import numpy as np

from swcgeom.core import Tree
from swcgeom.core.swc_utils import SWCNames, SWCTypes, get_names, get_types
from swcgeom.transforms.base import Transform

__all__ = ["NeurolucidaAscToSwc"]


class NeurolucidaAscToSwc(Transform[str, Tree]):
    """Convert neurolucida asc format to swc format."""

    def __call__(self, x: str) -> Tree:
        return self.convert(x)

    @classmethod
    def convert(cls, fname: str) -> Tree:
        with open(fname, "r") as f:
            tree = cls.from_stream(f, source=os.path.abspath(fname))

        return tree

    @classmethod
    def from_stream(cls, x: TextIOBase, *, source: str = "") -> Tree:
        parser = Parser(x, source=source)
        ast = parser.parse()
        tree = cls.from_ast(ast)
        return tree

    @staticmethod
    def from_ast(
        ast: "AST",
        *,
        names: Optional[SWCNames] = None,
        types: Optional[SWCTypes] = None,
    ) -> Tree:
        names = get_names(names)
        types = get_types(types)
        ndata = {n: [] for n in names.cols()}

        next_id = 0
        typee = [types.undefined]

        def walk_ast(root: ASTNode, pid: int = -1) -> None:
            nonlocal next_id, typee
            match root.type:
                case ASTType.ROOT:
                    for n in root.children:
                        walk_ast(n)

                case ASTType.TREE:
                    match root.value:
                        case "AXON":
                            typee.append(types.axon)
                        case "DENDRITE":
                            typee.append(types.basal_dendrite)

                    for n in root.children:
                        walk_ast(n)

                    typee.pop()

                case ASTType.NODE:
                    x, y, z, r = root.value
                    idx = next_id
                    next_id += 1

                    ndata[names.id].append(idx)
                    ndata[names.type].append(typee[-1])
                    ndata[names.x].append(x)
                    ndata[names.y].append(y)
                    ndata[names.z].append(z)
                    ndata[names.r].append(r)
                    ndata[names.pid].append(pid)

                    for n in root.children:
                        walk_ast(n, pid=idx)

        walk_ast(ast)
        tree = Tree(
            next_id,
            source=ast.source,
            names=names,
            **ndata,  # type: ignore
        )
        return tree


# -----------------
# ASC format parser
# -----------------

# AST


class ASTType(Enum):
    ROOT = auto()
    TREE = auto()
    NODE = auto()
    COLOR = auto()
    COMMENT = auto()


class ASTNode:
    parent: "ASTNode | None" = None

    def __init__(
        self,
        type: ASTType,
        value: Any = None,
        tokens: Optional[List["Token"]] = None,
        children: Optional[List["ASTNode"]] = None,
    ):
        self.type = type
        self.value = value
        self.tokens = tokens or []
        self.children = children or []
        for child in self.children:
            child.parent = self

    def add_child(self, child: "ASTNode") -> None:
        self.children.append(child)
        child.parent = self
        if child.tokens is not None:
            self.tokens.extend(child.tokens)

    def __eq__(self, __value: object) -> bool:
        """
        Compare two ASTNode objects.

        Notes
        -----
        The `parent`, `tokens` attribute is not compared.
        """
        return (
            isinstance(__value, ASTNode)
            and self.type == __value.type
            and self.value == __value.value
            and self.children == __value.children
        )


class AST(ASTNode):
    def __init__(self, children: Optional[List[ASTNode]] = None, source: str = ""):
        super().__init__(ASTType.ROOT, children=children)
        self.source = source


# ASC values


class ASCNode(NamedTuple):
    x: float
    y: float
    z: float
    r: float


class ASCColor(NamedTuple):
    color: str

    def __eq__(self, __value: object) -> bool:
        return (
            isinstance(__value, ASCColor)
            and self.color.upper() == __value.color.upper()
        )


class ASCComment(NamedTuple):
    comment: str


# Error


class TokenTypeError(ValueError):
    def __init__(self, token: "Token", expected: str):
        super().__init__(
            f"Unexpected token {token.type.name} `{token.value}` at {token.lineno}:{token.column}, expected {expected}"
        )


class LiteralTokenError(ValueError):
    def __init__(self, token: "Token", expected: str):
        super().__init__(
            f"Unexpected LITERAL token {token.value} at {token.lineno}:{token.column}, expected {expected}"
        )


class AssertionTokenTypeError(Exception):
    pass


# Parser


class Parser:
    def __init__(self, r: TextIOBase, *, source: str = ""):
        self.lexer = Lexer(r)
        self.next_token = None
        self.source = source
        self._read_token()

    def parse(self) -> AST:
        try:
            return self._parse()
        except AssertionTokenTypeError as assertion_err:
            msg = (
                f"Error parsing {self.source}" if self.source != "" else "Error parsing"
            )
            original_error = assertion_err.__cause__
            err = ValueError(msg)
            if original_error is None:
                raise err

            ignores = ["_assert_and_cunsume", "_assert"]
            current = assertion_err.__traceback__
            while current is not None:
                if (
                    current.tb_next is not None
                    and current.tb_next.tb_frame.f_code.co_name in ignores
                ):
                    current.tb_next = None
                else:
                    current = current.tb_next

            original_error.__traceback__ = assertion_err.__traceback__

            raise err from original_error
        except Exception as original_error:
            msg = f"Error parsing {self.source}" if self.source else "Error parsing"
            raise ValueError(msg) from original_error

    def _parse(self) -> AST:
        root = AST(source=self.source)

        token = self._assert_and_cunsume(TokenType.BRACKET_LEFT)
        root.tokens.append(token)

        while (token := self.next_token) is not None:
            if token.type == TokenType.BRACKET_RIGHT:
                break

            if token.type != TokenType.BRACKET_LEFT:
                raise TokenTypeError(token, "BRACKET_LEFT, BRACKET_RIGHT")

            root.tokens.append(token)
            self._consume()

            token = self._assert(self.next_token, TokenType.LITERAL)
            match str.upper(token.value):
                case "AXON" | "DENDRITE":
                    self._parse_tree(root)

                case "COLOR":
                    self._parse_color(root)  # TODO: bug

                case _:
                    raise LiteralTokenError(token, "AXON, DENDRITE, COLOR")

        token = self._assert(self.next_token, TokenType.BRACKET_RIGHT)
        token = self._assert_and_cunsume(TokenType.BRACKET_RIGHT)
        root.tokens.append(token)
        return root

    def _parse_tree(self, root: ASTNode) -> None:
        t1 = self._assert_and_cunsume(TokenType.LITERAL)
        node = ASTNode(ASTType.TREE, str.upper(t1.value), tokens=[t1])

        t2 = self._assert_and_cunsume(TokenType.BRACKET_RIGHT)
        node.tokens.append(t2)

        t3 = self._assert_and_cunsume(TokenType.BRACKET_LEFT)
        node.tokens.append(t3)

        self._parse_subtree(node)
        root.add_child(node)

    def _parse_subtree(self, root: ASTNode) -> None:
        flag = True  # flag to check if the brachet_left can be consumed
        current = root
        while (token := self.next_token) is not None:
            match token.type:
                case TokenType.BRACKET_LEFT:
                    self._read_token()
                    if flag:
                        flag = False
                    else:
                        self._parse_subtree(current)

                case TokenType.BRACKET_RIGHT:
                    break

                case TokenType.FLOAT:
                    current = self._parse_node(current)
                    flag = True

                case TokenType.LITERAL:
                    match str.upper(token.value):
                        case "COLOR":
                            self._parse_color(current)
                        case _:
                            raise LiteralTokenError(token, "COLOR")

                    flag = True

                case TokenType.OR:
                    current = root
                    self._read_token()
                    flag = True

                case TokenType.COMMENT:
                    self._parse_comment(current)

                case _:
                    excepted = (
                        "BRACKET_LEFT, BRACKET_RIGHT, LITERAL, FLOAT, OR, COMMENT"
                    )
                    raise TokenTypeError(token, excepted)

            current.tokens.append(token)

    def _parse_node(self, root: ASTNode) -> ASTNode:
        # FLOAT FLOAT FLOAT FLOAT )
        t1 = self._assert_and_cunsume(TokenType.FLOAT)
        t2 = self._assert(self.next_token, TokenType.FLOAT)
        self._read_token()
        t3 = self._assert(self.next_token, TokenType.FLOAT)
        self._read_token()
        t4 = self._assert(self.next_token, TokenType.FLOAT)
        self._read_token()
        t5 = self._assert_and_cunsume(TokenType.BRACKET_RIGHT)

        x, y, z, r = t1.value, t2.value, t3.value, t4.value
        node = ASTNode(ASTType.NODE, ASCNode(x, y, z, r), tokens=[t1, t2, t3, t4, t5])
        root.add_child(node)
        return node

    def _parse_color(self, root: ASTNode) -> ASTNode:
        # COLOR COLOR_VALUE )
        t1 = self._assert_and_cunsume(TokenType.LITERAL)
        t2 = self._assert_and_cunsume(TokenType.LITERAL)
        t3 = self._assert_and_cunsume(TokenType.BRACKET_RIGHT)

        node = ASTNode(ASTType.COLOR, ASCColor(t2.value), tokens=[t1, t2, t3])
        root.add_child(node)
        return node

    def _parse_comment(self, root: ASTNode) -> ASTNode:
        # ; COMMENT
        t1 = self._assert_and_cunsume(TokenType.COMMENT)
        node = ASTNode(ASTType.COMMENT, ASCComment(t1.value), tokens=[t1])
        root.add_child(node)  # ? where the comment should be added
        return node

    def _read_token(self) -> None:
        self.next_token = next(self.lexer, None)

    def _assert_and_cunsume(self, type: "TokenType") -> "Token":
        token = self._consume()
        token = self._assert(token, type)
        return cast(Token, token)

    def _assert(self, token: "Token | None", type: "TokenType") -> "Token":
        if token is None:
            raise AssertionTokenTypeError() from ValueError("Unexpected EOF")

        if token.type != type:
            raise AssertionTokenTypeError() from TokenTypeError(token, type.name)

        return token

    def _consume(self) -> "Token | None":
        token = self.next_token
        self._read_token()
        return token


# -----------------
# ASC format lexer
# -----------------


class TokenType(Enum):
    BRACKET_LEFT = auto()
    BRACKET_RIGHT = auto()
    COMMENT = auto()
    OR = auto()
    FLOAT = auto()
    LITERAL = auto()


class Token:
    def __init__(self, type: TokenType, value: Any, lineno: int, column: int):
        self.type = type
        self.value = value
        self.lineno = lineno
        self.column = column

    def __repr__(self) -> str:
        return f"Token({self.type.name}, {self.value}, Position={self.lineno}:{self.column})"


RE_FLOAT = re.compile(r"[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?")


class Lexer:
    def __init__(self, r: TextIOBase):
        self.r = r
        self.lineno = 1
        self.column = 1
        self.next_char = self.r.read(1)

    def __iter__(self):
        return self

    def __next__(self) -> Token:
        match (word := self._read_word()):
            case "":
                raise StopIteration

            case "(":
                return self._token(TokenType.BRACKET_LEFT, word)

            case ")":
                return self._token(TokenType.BRACKET_RIGHT, word)

            case ";":
                return self._token(TokenType.COMMENT, self._read_line())

            case "|":
                return self._token(TokenType.OR, word)

            case _ if RE_FLOAT.match(word) is not None:
                return self._token(TokenType.FLOAT, float(word))

            case _:
                return self._token(TokenType.LITERAL, word)

    def _read_char(self) -> bool:
        self.next_char = self.r.read(1)
        if self.next_char == "":
            return False

        if self.next_char == "\n":
            self.lineno += 1
            self.column = 1
        else:
            self.column += 1
        return True

    def _read_word(self) -> str:
        # skip leading spaces
        while self.next_char != "" and self.next_char in " \t\n":
            self._read_char()

        token = ""
        while self.next_char != "" and self.next_char not in " \t\n();|":
            token += self.next_char
            self._read_char()

        if token != "":
            return token

        if self.next_char == "":
            return ""

        ch = self.next_char
        self._read_char()
        return ch

    def _read_line(self) -> str:
        if self.next_char != "\n":
            line = self.r.readline()
            line = self.next_char + line
            if line.endswith("\n"):
                line = line[:-1]
        else:
            line = ""

        self.lineno += 1
        self.column = 1
        self.next_char = self.r.read(1)
        return line

    def _token(self, type: TokenType, value: Any) -> Token:
        return Token(type, value, self.lineno, self.column)
