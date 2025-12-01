import re
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field


class TokenType(Enum):
    KEYWORD = "KEYWORD"
    IDENTIFIER = "IDENTIFIER"
    NUMBER = "NUMBER"
    OPERATOR = "OPERATOR"
    DELIMITER = "DELIMITER"
    STRING = "STRING"
    COMMENT = "COMMENT"
    EOF = "EOF"


@dataclass
class Token:
    type: TokenType
    value: str
    line: int
    column: int


class Lexer:
    def __init__(self, source: str):
        self.source = source
        self.position = 0
        self.line = 1
        self.column = 1
        self.tokens = []
        self.keywords = {'if', 'else', 'while', 'for', 'int', 'float', 'return', 'void', 'string'}
        self.operators = {'+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!'}
        self.delimiters = {';', ',', '(', ')', '{', '}', '[', ']'}
    
    def current_char(self) -> Optional[str]:
        if self.position >= len(self.source):
            return None
        return self.source[self.position]
    
    def peek_char(self, offset: int = 1) -> Optional[str]:
        pos = self.position + offset
        if pos >= len(self.source):
            return None
        return self.source[pos]
    
    def advance(self):
        if self.position < len(self.source):
            if self.source[self.position] == '\n':
                self.line += 1
                self.column = 1
            else:
                self.column += 1
            self.position += 1
    
    def skip_whitespace(self):
        while self.current_char() and self.current_char() in ' \t\n\r':
            self.advance()
    
    def read_number(self) -> Token:
        start_line = self.line
        start_column = self.column
        num_str = ''
        
        while self.current_char() and (self.current_char().isdigit() or self.current_char() == '.'):
            num_str += self.current_char()
            self.advance()
        
        return Token(TokenType.NUMBER, num_str, start_line, start_column)
    
    def read_identifier(self) -> Token:
        start_line = self.line
        start_column = self.column
        id_str = ''
        
        while self.current_char() and (self.current_char().isalnum() or self.current_char() == '_'):
            id_str += self.current_char()
            self.advance()
        
        token_type = TokenType.KEYWORD if id_str in self.keywords else TokenType.IDENTIFIER
        return Token(token_type, id_str, start_line, start_column)
    
    def read_string(self) -> Token:
        start_line = self.line
        start_column = self.column
        self.advance()
        str_value = ''
        
        while self.current_char() and self.current_char() != '"':
            str_value += self.current_char()
            self.advance()
        
        if self.current_char() == '"':
            self.advance()
        
        return Token(TokenType.STRING, str_value, start_line, start_column)
    
    def read_operator(self) -> Token:
        start_line = self.line
        start_column = self.column
        op = self.current_char()
        self.advance()
        
        if self.current_char() and op + self.current_char() in self.operators:
            op += self.current_char()
            self.advance()
        
        return Token(TokenType.OPERATOR, op, start_line, start_column)
    
    def tokenize(self) -> List[Token]:
        while self.current_char():
            self.skip_whitespace()
            
            if not self.current_char():
                break
            
            char = self.current_char()
            
            if char.isdigit():
                self.tokens.append(self.read_number())
            elif char.isalpha() or char == '_':
                self.tokens.append(self.read_identifier())
            elif char == '"':
                self.tokens.append(self.read_string())
            elif char in self.delimiters:
                self.tokens.append(Token(TokenType.DELIMITER, char, self.line, self.column))
                self.advance()
            elif char in {'+', '-', '*', '/', '=', '!', '<', '>', '&', '|'}:
                self.tokens.append(self.read_operator())
            else:
                self.advance()
        
        self.tokens.append(Token(TokenType.EOF, '', self.line, self.column))
        return self.tokens


@dataclass
class ASTNode:
    node_type: str
    value: Any = None
    children: List['ASTNode'] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)


class Parser:
    def __init__(self, tokens: List[Token]):
        self.tokens = tokens
        self.position = 0
        self.errors = []
    
    def current_token(self) -> Token:
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return self.tokens[-1]
    
    def peek_token(self, offset: int = 1) -> Token:
        pos = self.position + offset
        if pos < len(self.tokens):
            return self.tokens[pos]
        return self.tokens[-1]
    
    def advance(self):
        if self.position < len(self.tokens) - 1:
            self.position += 1
    
    def expect(self, token_type: TokenType, value: Optional[str] = None) -> bool:
        token = self.current_token()
        if token.type == token_type and (value is None or token.value == value):
            self.advance()
            return True
        self.errors.append(f"Expected {token_type.value} {value or ''} at line {token.line}")
        return False
    
    def parse(self) -> ASTNode:
        root = ASTNode("Program")
        
        while self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                root.children.append(stmt)
        
        return root
    
    def parse_statement(self) -> Optional[ASTNode]:
        token = self.current_token()
        
        if token.type == TokenType.KEYWORD:
            if token.value in {'int', 'float', 'string', 'void'}:
                return self.parse_declaration()
            elif token.value == 'if':
                return self.parse_if_statement()
            elif token.value == 'while':
                return self.parse_while_statement()
            elif token.value == 'for':
                return self.parse_for_statement()
            elif token.value == 'return':
                return self.parse_return_statement()
        
        return self.parse_expression_statement()
    
    def parse_declaration(self) -> ASTNode:
        type_token = self.current_token()
        self.advance()
        
        name_token = self.current_token()
        node = ASTNode("Declaration", attributes={'type': type_token.value, 'name': name_token.value})
        
        if name_token.type == TokenType.IDENTIFIER:
            self.advance()
            
            if self.current_token().value == '(':
                return self.parse_function_declaration(type_token.value, name_token.value)
            
            if self.current_token().value == '=':
                self.advance()
                node.children.append(self.parse_expression())
            
            self.expect(TokenType.DELIMITER, ';')
        
        return node
    
    def parse_function_declaration(self, return_type: str, name: str) -> ASTNode:
        node = ASTNode("FunctionDeclaration", attributes={'type': return_type, 'name': name})
        
        self.expect(TokenType.DELIMITER, '(')
        
        params = []
        while self.current_token().value != ')':
            if self.current_token().type == TokenType.KEYWORD:
                param_type = self.current_token().value
                self.advance()
                param_name = self.current_token().value
                self.advance()
                params.append({'type': param_type, 'name': param_name})
                
                if self.current_token().value == ',':
                    self.advance()
        
        node.attributes['params'] = params
        self.expect(TokenType.DELIMITER, ')')
        
        if self.current_token().value == '{':
            node.children.append(self.parse_block())
        
        return node
    
    def parse_block(self) -> ASTNode:
        node = ASTNode("Block")
        self.expect(TokenType.DELIMITER, '{')
        
        while self.current_token().value != '}' and self.current_token().type != TokenType.EOF:
            stmt = self.parse_statement()
            if stmt:
                node.children.append(stmt)
        
        self.expect(TokenType.DELIMITER, '}')
        return node
    
    def parse_if_statement(self) -> ASTNode:
        node = ASTNode("IfStatement")
        self.advance()
        
        self.expect(TokenType.DELIMITER, '(')
        node.children.append(self.parse_expression())
        self.expect(TokenType.DELIMITER, ')')
        
        node.children.append(self.parse_block())
        
        if self.current_token().value == 'else':
            self.advance()
            node.children.append(self.parse_block())
        
        return node
    
    def parse_while_statement(self) -> ASTNode:
        node = ASTNode("WhileStatement")
        self.advance()
        
        self.expect(TokenType.DELIMITER, '(')
        node.children.append(self.parse_expression())
        self.expect(TokenType.DELIMITER, ')')
        
        node.children.append(self.parse_block())
        
        return node
    
    def parse_for_statement(self) -> ASTNode:
        node = ASTNode("ForStatement")
        self.advance()
        
        self.expect(TokenType.DELIMITER, '(')
        node.children.append(self.parse_statement())
        node.children.append(self.parse_expression())
        self.expect(TokenType.DELIMITER, ';')
        node.children.append(self.parse_expression())
        self.expect(TokenType.DELIMITER, ')')
        
        node.children.append(self.parse_block())
        
        return node
    
    def parse_return_statement(self) -> ASTNode:
        node = ASTNode("ReturnStatement")
        self.advance()
        
        if self.current_token().value != ';':
            node.children.append(self.parse_expression())
        
        self.expect(TokenType.DELIMITER, ';')
        return node
    
    def parse_expression_statement(self) -> ASTNode:
        node = self.parse_expression()
        self.expect(TokenType.DELIMITER, ';')
        return node
    
    def parse_expression(self) -> ASTNode:
        return self.parse_assignment()
    
    def parse_assignment(self) -> ASTNode:
        node = self.parse_logical_or()
        
        if self.current_token().value == '=':
            op_token = self.current_token()
            self.advance()
            right = self.parse_assignment()
            return ASTNode("Assignment", op_token.value, [node, right])
        
        return node
    
    def parse_logical_or(self) -> ASTNode:
        node = self.parse_logical_and()
        
        while self.current_token().value == '||':
            op_token = self.current_token()
            self.advance()
            right = self.parse_logical_and()
            node = ASTNode("BinaryOp", op_token.value, [node, right])
        
        return node
    
    def parse_logical_and(self) -> ASTNode:
        node = self.parse_equality()
        
        while self.current_token().value == '&&':
            op_token = self.current_token()
            self.advance()
            right = self.parse_equality()
            node = ASTNode("BinaryOp", op_token.value, [node, right])
        
        return node
    
    def parse_equality(self) -> ASTNode:
        node = self.parse_relational()
        
        while self.current_token().value in {'==', '!='}:
            op_token = self.current_token()
            self.advance()
            right = self.parse_relational()
            node = ASTNode("BinaryOp", op_token.value, [node, right])
        
        return node
    
    def parse_relational(self) -> ASTNode:
        node = self.parse_additive()
        
        while self.current_token().value in {'<', '>', '<=', '>='}:
            op_token = self.current_token()
            self.advance()
            right = self.parse_additive()
            node = ASTNode("BinaryOp", op_token.value, [node, right])
        
        return node
    
    def parse_additive(self) -> ASTNode:
        node = self.parse_multiplicative()
        
        while self.current_token().value in {'+', '-'}:
            op_token = self.current_token()
            self.advance()
            right = self.parse_multiplicative()
            node = ASTNode("BinaryOp", op_token.value, [node, right])
        
        return node
    
    def parse_multiplicative(self) -> ASTNode:
        node = self.parse_unary()
        
        while self.current_token().value in {'*', '/'}:
            op_token = self.current_token()
            self.advance()
            right = self.parse_unary()
            node = ASTNode("BinaryOp", op_token.value, [node, right])
        
        return node
    
    def parse_unary(self) -> ASTNode:
        if self.current_token().value in {'-', '!'}:
            op_token = self.current_token()
            self.advance()
            return ASTNode("UnaryOp", op_token.value, [self.parse_unary()])
        
        return self.parse_primary()
    
    def parse_primary(self) -> ASTNode:
        token = self.current_token()
        
        if token.type == TokenType.NUMBER:
            self.advance()
            return ASTNode("Literal", token.value, attributes={'type': 'number'})
        
        elif token.type == TokenType.STRING:
            self.advance()
            return ASTNode("Literal", token.value, attributes={'type': 'string'})
        
        elif token.type == TokenType.IDENTIFIER:
            name = token.value
            self.advance()
            
            if self.current_token().value == '(':
                return self.parse_function_call(name)
            
            return ASTNode("Identifier", name)
        
        elif token.value == '(':
            self.advance()
            node = self.parse_expression()
            self.expect(TokenType.DELIMITER, ')')
            return node
        
        return ASTNode("Error", "Unexpected token")
    
    def parse_function_call(self, name: str) -> ASTNode:
        node = ASTNode("FunctionCall", name)
        self.expect(TokenType.DELIMITER, '(')
        
        while self.current_token().value != ')' and self.current_token().type != TokenType.EOF:
            node.children.append(self.parse_expression())
            
            if self.current_token().value == ',':
                self.advance()
        
        self.expect(TokenType.DELIMITER, ')')
        return node


class Symbol:
    def __init__(self, name: str, symbol_type: str, scope: int, value_type: Optional[str] = None):
        self.name = name
        self.symbol_type = symbol_type
        self.scope = scope
        self.value_type = value_type
        self.is_initialized = False


class SymbolTable:
    def __init__(self):
        self.scopes = [{}]
        self.current_scope = 0
    
    def enter_scope(self):
        self.current_scope += 1
        if len(self.scopes) <= self.current_scope:
            self.scopes.append({})
    
    def exit_scope(self):
        if self.current_scope > 0:
            self.current_scope -= 1
    
    def declare(self, name: str, symbol_type: str, value_type: Optional[str] = None) -> bool:
        if name in self.scopes[self.current_scope]:
            return False
        
        self.scopes[self.current_scope][name] = Symbol(name, symbol_type, self.current_scope, value_type)
        return True
    
    def lookup(self, name: str) -> Optional[Symbol]:
        for i in range(self.current_scope, -1, -1):
            if name in self.scopes[i]:
                return self.scopes[i][name]
        return None
    
    def update_initialization(self, name: str):
        symbol = self.lookup(name)
        if symbol:
            symbol.is_initialized = True


class SemanticAnalyzer:
    def __init__(self, ast: ASTNode):
        self.ast = ast
        self.symbol_table = SymbolTable()
        self.errors = []
        self.warnings = []
        self.semantic_tree = None
        self.type_map = {
            'int': 'int',
            'float': 'float',
            'string': 'string',
            'void': 'void'
        }
    
    def analyze(self) -> Tuple[ASTNode, List[str], List[str]]:
        self.semantic_tree = self.annotate_node(self.ast)
        return self.semantic_tree, self.errors, self.warnings
    
    def annotate_node(self, node: ASTNode) -> ASTNode:
        if node.node_type == "Program":
            annotated = ASTNode("Program", attributes={'scope': 0})
            for child in node.children:
                annotated.children.append(self.annotate_node(child))
            return annotated
        
        elif node.node_type == "Declaration":
            return self.analyze_declaration(node)
        
        elif node.node_type == "FunctionDeclaration":
            return self.analyze_function_declaration(node)
        
        elif node.node_type == "Assignment":
            return self.analyze_assignment(node)
        
        elif node.node_type == "BinaryOp":
            return self.analyze_binary_op(node)
        
        elif node.node_type == "UnaryOp":
            return self.analyze_unary_op(node)
        
        elif node.node_type == "Identifier":
            return self.analyze_identifier(node)
        
        elif node.node_type == "Literal":
            return self.analyze_literal(node)
        
        elif node.node_type == "FunctionCall":
            return self.analyze_function_call(node)
        
        elif node.node_type == "IfStatement":
            return self.analyze_if_statement(node)
        
        elif node.node_type == "WhileStatement":
            return self.analyze_while_statement(node)
        
        elif node.node_type == "ForStatement":
            return self.analyze_for_statement(node)
        
        elif node.node_type == "ReturnStatement":
            return self.analyze_return_statement(node)
        
        elif node.node_type == "Block":
            return self.analyze_block(node)
        
        else:
            annotated = ASTNode(node.node_type, node.value, attributes=node.attributes.copy())
            for child in node.children:
                annotated.children.append(self.annotate_node(child))
            return annotated
    
    def analyze_declaration(self, node: ASTNode) -> ASTNode:
        var_type = node.attributes['type']
        var_name = node.attributes['name']
        
        if not self.symbol_table.declare(var_name, 'variable', var_type):
            self.errors.append(f"Variable '{var_name}' already declared in this scope")
        
        annotated = ASTNode("Declaration", attributes={
            'type': var_type,
            'name': var_name,
            'scope': self.symbol_table.current_scope,
            'semantic_type': var_type
        })
        
        if node.children:
            init_expr = self.annotate_node(node.children[0])
            annotated.children.append(init_expr)
            
            expr_type = init_expr.attributes.get('semantic_type')
            if expr_type and not self.types_compatible(var_type, expr_type):
                self.errors.append(f"Type mismatch: cannot assign {expr_type} to {var_type}")
            
            self.symbol_table.update_initialization(var_name)
        else:
            self.warnings.append(f"Variable '{var_name}' declared but not initialized")
        
        return annotated
    
    def analyze_function_declaration(self, node: ASTNode) -> ASTNode:
        func_type = node.attributes['type']
        func_name = node.attributes['name']
        params = node.attributes.get('params', [])
        
        if not self.symbol_table.declare(func_name, 'function', func_type):
            self.errors.append(f"Function '{func_name}' already declared")
        
        self.symbol_table.enter_scope()
        
        for param in params:
            self.symbol_table.declare(param['name'], 'parameter', param['type'])
        
        annotated = ASTNode("FunctionDeclaration", attributes={
            'type': func_type,
            'name': func_name,
            'params': params,
            'scope': self.symbol_table.current_scope - 1,
            'semantic_type': func_type
        })
        
        for child in node.children:
            annotated.children.append(self.annotate_node(child))
        
        self.symbol_table.exit_scope()
        
        return annotated
    
    def analyze_assignment(self, node: ASTNode) -> ASTNode:
        left = self.annotate_node(node.children[0])
        right = self.annotate_node(node.children[1])
        
        left_type = left.attributes.get('semantic_type')
        right_type = right.attributes.get('semantic_type')
        
        if left_type and right_type and not self.types_compatible(left_type, right_type):
            self.errors.append(f"Type mismatch in assignment: {left_type} = {right_type}")
        
        if left.node_type == "Identifier":
            self.symbol_table.update_initialization(left.value)
        
        annotated = ASTNode("Assignment", node.value, [left, right], attributes={
            'semantic_type': left_type or 'unknown'
        })
        
        return annotated
    
    def analyze_binary_op(self, node: ASTNode) -> ASTNode:
        left = self.annotate_node(node.children[0])
        right = self.annotate_node(node.children[1])
        
        left_type = left.attributes.get('semantic_type')
        right_type = right.attributes.get('semantic_type')
        
        result_type = self.infer_binary_type(node.value, left_type, right_type)
        
        if result_type == 'error':
            self.errors.append(f"Invalid operation {node.value} between {left_type} and {right_type}")
            result_type = 'unknown'
        
        annotated = ASTNode("BinaryOp", node.value, [left, right], attributes={
            'semantic_type': result_type
        })
        
        return annotated
    
    def analyze_unary_op(self, node: ASTNode) -> ASTNode:
        operand = self.annotate_node(node.children[0])
        operand_type = operand.attributes.get('semantic_type')
        
        if node.value == '-' and operand_type not in {'int', 'float'}:
            self.errors.append(f"Unary minus requires numeric type, got {operand_type}")
        
        if node.value == '!' and operand_type not in {'int', 'bool'}:
            self.errors.append(f"Logical NOT requires boolean type, got {operand_type}")
        
        annotated = ASTNode("UnaryOp", node.value, [operand], attributes={
            'semantic_type': operand_type or 'unknown'
        })
        
        return annotated
    
    def analyze_identifier(self, node: ASTNode) -> ASTNode:
        symbol = self.symbol_table.lookup(node.value)
        
        if not symbol:
            self.errors.append(f"Undefined identifier '{node.value}'")
            return ASTNode("Identifier", node.value, attributes={'semantic_type': 'unknown'})
        
        if not symbol.is_initialized:
            self.warnings.append(f"Variable '{node.value}' used before initialization")
        
        return ASTNode("Identifier", node.value, attributes={
            'semantic_type': symbol.value_type or 'unknown',
            'scope': symbol.scope,
            'symbol_type': symbol.symbol_type
        })
    
    def analyze_literal(self, node: ASTNode) -> ASTNode:
        lit_type = node.attributes.get('type', 'unknown')
        
        if lit_type == 'number':
            if '.' in str(node.value):
                semantic_type = 'float'
            else:
                semantic_type = 'int'
        else:
            semantic_type = lit_type
        
        return ASTNode("Literal", node.value, attributes={
            'type': lit_type,
            'semantic_type': semantic_type
        })
    
    def analyze_function_call(self, node: ASTNode) -> ASTNode:
        func_name = node.value
        symbol = self.symbol_table.lookup(func_name)
        
        if not symbol:
            self.errors.append(f"Undefined function '{func_name}'")
            return ASTNode("FunctionCall", func_name, attributes={'semantic_type': 'unknown'})
        
        if symbol.symbol_type != 'function':
            self.errors.append(f"'{func_name}' is not a function")
        
        annotated = ASTNode("FunctionCall", func_name, attributes={
            'semantic_type': symbol.value_type or 'unknown',
            'scope': symbol.scope
        })
        
        for child in node.children:
            annotated.children.append(self.annotate_node(child))
        
        return annotated
    
    def analyze_if_statement(self, node: ASTNode) -> ASTNode:
        condition = self.annotate_node(node.children[0])
        
        annotated = ASTNode("IfStatement", attributes={'scope': self.symbol_table.current_scope})
        annotated.children.append(condition)
        
        self.symbol_table.enter_scope()
        then_block = self.annotate_node(node.children[1])
        annotated.children.append(then_block)
        self.symbol_table.exit_scope()
        
        if len(node.children) > 2:
            self.symbol_table.enter_scope()
            else_block = self.annotate_node(node.children[2])
            annotated.children.append(else_block)
            self.symbol_table.exit_scope()
        
        return annotated
    
    def analyze_while_statement(self, node: ASTNode) -> ASTNode:
        condition = self.annotate_node(node.children[0])
        
        annotated = ASTNode("WhileStatement", attributes={'scope': self.symbol_table.current_scope})
        annotated.children.append(condition)
        
        self.symbol_table.enter_scope()
        body = self.annotate_node(node.children[1])
        annotated.children.append(body)
        self.symbol_table.exit_scope()
        
        return annotated
    
    def analyze_for_statement(self, node: ASTNode) -> ASTNode:
        annotated = ASTNode("ForStatement", attributes={'scope': self.symbol_table.current_scope})
        
        self.symbol_table.enter_scope()
        
        for child in node.children:
            annotated.children.append(self.annotate_node(child))
        
        self.symbol_table.exit_scope()
        
        return annotated
    
    def analyze_return_statement(self, node: ASTNode) -> ASTNode:
        annotated = ASTNode("ReturnStatement", attributes={'scope': self.symbol_table.current_scope})
        
        if node.children:
            return_expr = self.annotate_node(node.children[0])
            annotated.children.append(return_expr)
            annotated.attributes['semantic_type'] = return_expr.attributes.get('semantic_type')
        else:
            annotated.attributes['semantic_type'] = 'void'
        
        return annotated
    
    def analyze_block(self, node: ASTNode) -> ASTNode:
        annotated = ASTNode("Block", attributes={'scope': self.symbol_table.current_scope})
        
        for child in node.children:
            annotated.children.append(self.annotate_node(child))
        
        return annotated
    
    def types_compatible(self, type1: str, type2: str) -> bool:
        if type1 == type2:
            return True
        
        if type1 in {'int', 'float'} and type2 in {'int', 'float'}:
            return True
        
        return False
    
    def infer_binary_type(self, op: str, left_type: Optional[str], right_type: Optional[str]) -> str:
        if not left_type or not right_type:
            return 'unknown'
        
        if op in {'+', '-', '*', '/'}:
            if left_type in {'int', 'float'} and right_type in {'int', 'float'}:
                if left_type == 'float' or right_type == 'float':
                    return 'float'
                return 'int'
            return 'error'
        
        if op in {'==', '!=', '<', '>', '<=', '>='}:
            if self.types_compatible(left_type, right_type):
                return 'int'
            return 'error'
        
        if op in {'&&', '||'}:
            return 'int'
        
        return 'unknown'


def ast_to_dict(node: ASTNode) -> Dict:
    return {
        'node_type': node.node_type,
        'value': node.value,
        'attributes': node.attributes,
        'children': [ast_to_dict(child) for child in node.children]
    }


def analyze_source_code(source_code: str) -> Dict[str, Any]:
    lexer = Lexer(source_code)
    tokens = lexer.tokenize()
    
    tokens_data = [
        {
            'type': token.type.value,
            'value': token.value,
            'line': token.line,
            'column': token.column
        }
        for token in tokens
    ]
    
    parser = Parser(tokens)
    syntax_tree = parser.parse()
    
    semantic_analyzer = SemanticAnalyzer(syntax_tree)
    semantic_tree, semantic_errors, semantic_warnings = semantic_analyzer.analyze()
    
    all_errors = parser.errors + semantic_errors
    all_warnings = semantic_warnings
    
    return {
        'tokens': tokens_data,
        'syntax_tree': ast_to_dict(syntax_tree),
        'semantic_tree': ast_to_dict(semantic_tree),
        'errors': all_errors,
        'warnings': all_warnings,
        'success': len(all_errors) == 0
    }


def main():
    sample_code = """
int x = 10;
float y = 20.5;
int z = x + y;

int add(int a, int b) {
    return a + b;
}

int result = add(x, z);

if (result > 20) {
    int temp = result * 2;
}

while (x < 100) {
    x = x + 1;
}
"""
    
    print("Compiler Analysis System")
    print("=" * 50)
    print("\nSource Code:")
    print(sample_code)
    print("\n" + "=" * 50)
    
    result = analyze_source_code(sample_code)
    
    print("\n--- LEXICAL ANALYSIS ---")
    print(f"Total Tokens: {len(result['tokens'])}")
    for token in result['tokens'][:20]:
        print(f"  {token['type']:15} | {token['value']:15} | Line {token['line']}, Col {token['column']}")
    
    print("\n--- SYNTAX ANALYSIS ---")
    print(f"Syntax Tree Root: {result['syntax_tree']['node_type']}")
    print(f"Top-level Statements: {len(result['syntax_tree']['children'])}")
    
    print("\n--- SEMANTIC ANALYSIS ---")
    print(f"Semantic Tree Root: {result['semantic_tree']['node_type']}")
    print(f"Analysis Success: {result['success']}")
    
    if result['errors']:
        print("\nErrors:")
        for error in result['errors']:
            print(f"  - {error}")
    
    if result['warnings']:
        print("\nWarnings:")
        for warning in result['warnings']:
            print(f"  - {warning}")
    
    print("\n" + "=" * 50)
    print("Analysis Complete!")


if __name__ == "__main__":
    main()
