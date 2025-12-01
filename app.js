class TokenType {
            static KEYWORD = "KEYWORD";
            static IDENTIFIER = "IDENTIFIER";
            static NUMBER = "NUMBER";
            static OPERATOR = "OPERATOR";
            static DELIMITER = "DELIMITER";
            static STRING = "STRING";
            static COMMENT = "COMMENT";
            static EOF = "EOF";
        }

        class Token {
            constructor(type, value, line, column) {
                this.type = type;
                this.value = value;
                this.line = line;
                this.column = column;
            }
        }

        class Lexer {
            constructor(source) {
                this.source = source;
                this.position = 0;
                this.line = 1;
                this.column = 1;
                this.tokens = [];
                this.keywords = new Set(['if', 'else', 'while', 'for', 'int', 'float', 'return', 'void', 'string']);
                this.operators = new Set(['+', '-', '*', '/', '=', '==', '!=', '<', '>', '<=', '>=', '&&', '||', '!']);
                this.delimiters = new Set([';', ',', '(', ')', '{', '}', '[', ']']);
            }

            currentChar() {
                if (this.position >= this.source.length) return null;
                return this.source[this.position];
            }

            peekChar(offset = 1) {
                const pos = this.position + offset;
                if (pos >= this.source.length) return null;
                return this.source[pos];
            }

            advance() {
                if (this.position < this.source.length) {
                    if (this.source[this.position] === '\n') {
                        this.line++;
                        this.column = 1;
                    } else {
                        this.column++;
                    }
                    this.position++;
                }
            }

            skipWhitespace() {
                while (this.currentChar() && ' \t\n\r'.includes(this.currentChar())) {
                    this.advance();
                }
            }

            readNumber() {
                const startLine = this.line;
                const startColumn = this.column;
                let numStr = '';

                while (this.currentChar() && (this.currentChar().match(/[0-9.]/) !== null)) {
                    numStr += this.currentChar();
                    this.advance();
                }

                return new Token(TokenType.NUMBER, numStr, startLine, startColumn);
            }

            readIdentifier() {
                const startLine = this.line;
                const startColumn = this.column;
                let idStr = '';

                while (this.currentChar() && (this.currentChar().match(/[a-zA-Z0-9_]/) !== null)) {
                    idStr += this.currentChar();
                    this.advance();
                }

                const tokenType = this.keywords.has(idStr) ? TokenType.KEYWORD : TokenType.IDENTIFIER;
                return new Token(tokenType, idStr, startLine, startColumn);
            }

            readString() {
                const startLine = this.line;
                const startColumn = this.column;
                this.advance();
                let strValue = '';

                while (this.currentChar() && this.currentChar() !== '"') {
                    strValue += this.currentChar();
                    this.advance();
                }

                if (this.currentChar() === '"') {
                    this.advance();
                }

                return new Token(TokenType.STRING, strValue, startLine, startColumn);
            }

            readOperator() {
                const startLine = this.line;
                const startColumn = this.column;
                let op = this.currentChar();
                this.advance();

                if (this.currentChar() && this.operators.has(op + this.currentChar())) {
                    op += this.currentChar();
                    this.advance();
                }

                return new Token(TokenType.OPERATOR, op, startLine, startColumn);
            }

            tokenize() {
                while (this.currentChar()) {
                    this.skipWhitespace();

                    if (!this.currentChar()) break;

                    const char = this.currentChar();

                    if (char.match(/[0-9]/)) {
                        this.tokens.push(this.readNumber());
                    } else if (char.match(/[a-zA-Z_]/)) {
                        this.tokens.push(this.readIdentifier());
                    } else if (char === '"') {
                        this.tokens.push(this.readString());
                    } else if (this.delimiters.has(char)) {
                        this.tokens.push(new Token(TokenType.DELIMITER, char, this.line, this.column));
                        this.advance();
                    } else if ('+-*/=!<>&|'.includes(char)) {
                        this.tokens.push(this.readOperator());
                    } else {
                        this.advance();
                    }
                }

                this.tokens.push(new Token(TokenType.EOF, '', this.line, this.column));
                return this.tokens;
            }
        }

        class ASTNode {
            constructor(nodeType, value = null, children = [], attributes = {}) {
                this.nodeType = nodeType;
                this.value = value;
                this.children = children;
                this.attributes = attributes;
            }
        }

        class Parser {
            constructor(tokens) {
                this.tokens = tokens;
                this.position = 0;
                this.errors = [];
            }

            currentToken() {
                if (this.position < this.tokens.length) {
                    return this.tokens[this.position];
                }
                return this.tokens[this.tokens.length - 1];
            }

            peekToken(offset = 1) {
                const pos = this.position + offset;
                if (pos < this.tokens.length) {
                    return this.tokens[pos];
                }
                return this.tokens[this.tokens.length - 1];
            }

            advance() {
                if (this.position < this.tokens.length - 1) {
                    this.position++;
                }
            }

            expect(tokenType, value = null) {
                const token = this.currentToken();
                if (token.type === tokenType && (value === null || token.value === value)) {
                    this.advance();
                    return true;
                }
                this.errors.push(`Expected ${tokenType} ${value || ''} at line ${token.line}`);
                return false;
            }

            parse() {
                const root = new ASTNode("Program");

                while (this.currentToken().type !== TokenType.EOF) {
                    const stmt = this.parseStatement();
                    if (stmt) {
                        root.children.push(stmt);
                    }
                }

                return root;
            }

            parseStatement() {
                const token = this.currentToken();

                if (token.type === TokenType.KEYWORD) {
                    if (['int', 'float', 'string', 'void'].includes(token.value)) {
                        return this.parseDeclaration();
                    } else if (token.value === 'if') {
                        return this.parseIfStatement();
                    } else if (token.value === 'while') {
                        return this.parseWhileStatement();
                    } else if (token.value === 'for') {
                        return this.parseForStatement();
                    } else if (token.value === 'return') {
                        return this.parseReturnStatement();
                    }
                }

                return this.parseExpressionStatement();
            }

            parseDeclaration() {
                const typeToken = this.currentToken();
                this.advance();

                const nameToken = this.currentToken();
                const node = new ASTNode("Declaration", null, [], {
                    type: typeToken.value,
                    name: nameToken.value
                });

                if (nameToken.type === TokenType.IDENTIFIER) {
                    this.advance();

                    if (this.currentToken().value === '(') {
                        return this.parseFunctionDeclaration(typeToken.value, nameToken.value);
                    }

                    if (this.currentToken().value === '=') {
                        this.advance();
                        node.children.push(this.parseExpression());
                    }

                    this.expect(TokenType.DELIMITER, ';');
                }

                return node;
            }

            parseFunctionDeclaration(returnType, name) {
                const node = new ASTNode("FunctionDeclaration", null, [], {
                    type: returnType,
                    name: name
                });

                this.expect(TokenType.DELIMITER, '(');

                const params = [];
                while (this.currentToken().value !== ')') {
                    if (this.currentToken().type === TokenType.KEYWORD) {
                        const paramType = this.currentToken().value;
                        this.advance();
                        const paramName = this.currentToken().value;
                        this.advance();
                        params.push({ type: paramType, name: paramName });

                        if (this.currentToken().value === ',') {
                            this.advance();
                        }
                    }
                }

                node.attributes.params = params;
                this.expect(TokenType.DELIMITER, ')');

                if (this.currentToken().value === '{') {
                    node.children.push(this.parseBlock());
                }

                return node;
            }

            parseBlock() {
                const node = new ASTNode("Block");
                this.expect(TokenType.DELIMITER, '{');

                while (this.currentToken().value !== '}' && this.currentToken().type !== TokenType.EOF) {
                    const stmt = this.parseStatement();
                    if (stmt) {
                        node.children.push(stmt);
                    }
                }

                this.expect(TokenType.DELIMITER, '}');
                return node;
            }

            parseIfStatement() {
                const node = new ASTNode("IfStatement");
                this.advance();

                this.expect(TokenType.DELIMITER, '(');
                node.children.push(this.parseExpression());
                this.expect(TokenType.DELIMITER, ')');

                node.children.push(this.parseBlock());

                if (this.currentToken().value === 'else') {
                    this.advance();
                    node.children.push(this.parseBlock());
                }

                return node;
            }

            parseWhileStatement() {
                const node = new ASTNode("WhileStatement");
                this.advance();

                this.expect(TokenType.DELIMITER, '(');
                node.children.push(this.parseExpression());
                this.expect(TokenType.DELIMITER, ')');

                node.children.push(this.parseBlock());

                return node;
            }

            parseForStatement() {
                const node = new ASTNode("ForStatement");
                this.advance();

                this.expect(TokenType.DELIMITER, '(');
                node.children.push(this.parseStatement());
                node.children.push(this.parseExpression());
                this.expect(TokenType.DELIMITER, ';');
                node.children.push(this.parseExpression());
                this.expect(TokenType.DELIMITER, ')');

                node.children.push(this.parseBlock());

                return node;
            }

            parseReturnStatement() {
                const node = new ASTNode("ReturnStatement");
                this.advance();

                if (this.currentToken().value !== ';') {
                    node.children.push(this.parseExpression());
                }

                this.expect(TokenType.DELIMITER, ';');
                return node;
            }

            parseExpressionStatement() {
                const node = this.parseExpression();
                this.expect(TokenType.DELIMITER, ';');
                return node;
            }

            parseExpression() {
                return this.parseAssignment();
            }

            parseAssignment() {
                let node = this.parseLogicalOr();

                if (this.currentToken().value === '=') {
                    const opToken = this.currentToken();
                    this.advance();
                    const right = this.parseAssignment();
                    return new ASTNode("Assignment", opToken.value, [node, right]);
                }

                return node;
            }

            parseLogicalOr() {
                let node = this.parseLogicalAnd();

                while (this.currentToken().value === '||') {
                    const opToken = this.currentToken();
                    this.advance();
                    const right = this.parseLogicalAnd();
                    node = new ASTNode("BinaryOp", opToken.value, [node, right]);
                }

                return node;
            }

            parseLogicalAnd() {
                let node = this.parseEquality();

                while (this.currentToken().value === '&&') {
                    const opToken = this.currentToken();
                    this.advance();
                    const right = this.parseEquality();
                    node = new ASTNode("BinaryOp", opToken.value, [node, right]);
                }

                return node;
            }

            parseEquality() {
                let node = this.parseRelational();

                while (['==', '!='].includes(this.currentToken().value)) {
                    const opToken = this.currentToken();
                    this.advance();
                    const right = this.parseRelational();
                    node = new ASTNode("BinaryOp", opToken.value, [node, right]);
                }

                return node;
            }

            parseRelational() {
                let node = this.parseAdditive();

                while (['<', '>', '<=', '>='].includes(this.currentToken().value)) {
                    const opToken = this.currentToken();
                    this.advance();
                    const right = this.parseAdditive();
                    node = new ASTNode("BinaryOp", opToken.value, [node, right]);
                }

                return node;
            }

            parseAdditive() {
                let node = this.parseMultiplicative();

                while (['+', '-'].includes(this.currentToken().value)) {
                    const opToken = this.currentToken();
                    this.advance();
                    const right = this.parseMultiplicative();
                    node = new ASTNode("BinaryOp", opToken.value, [node, right]);
                }

                return node;
            }

            parseMultiplicative() {
                let node = this.parseUnary();

                while (['*', '/'].includes(this.currentToken().value)) {
                    const opToken = this.currentToken();
                    this.advance();
                    const right = this.parseUnary();
                    node = new ASTNode("BinaryOp", opToken.value, [node, right]);
                }

                return node;
            }

            parseUnary() {
                if (['-', '!'].includes(this.currentToken().value)) {
                    const opToken = this.currentToken();
                    this.advance();
                    return new ASTNode("UnaryOp", opToken.value, [this.parseUnary()]);
                }

                return this.parsePrimary();
            }

            parsePrimary() {
                const token = this.currentToken();

                if (token.type === TokenType.NUMBER) {
                    this.advance();
                    return new ASTNode("Literal", token.value, [], { type: 'number' });
                } else if (token.type === TokenType.STRING) {
                    this.advance();
                    return new ASTNode("Literal", token.value, [], { type: 'string' });
                } else if (token.type === TokenType.IDENTIFIER) {
                    const name = token.value;
                    this.advance();

                    if (this.currentToken().value === '(') {
                        return this.parseFunctionCall(name);
                    }

                    return new ASTNode("Identifier", name);
                } else if (token.value === '(') {
                    this.advance();
                    const node = this.parseExpression();
                    this.expect(TokenType.DELIMITER, ')');
                    return node;
                }

                return new ASTNode("Error", "Unexpected token");
            }

            parseFunctionCall(name) {
                const node = new ASTNode("FunctionCall", name);
                this.expect(TokenType.DELIMITER, '(');

                while (this.currentToken().value !== ')' && this.currentToken().type !== TokenType.EOF) {
                    node.children.push(this.parseExpression());

                    if (this.currentToken().value === ',') {
                        this.advance();
                    }
                }

                this.expect(TokenType.DELIMITER, ')');
                return node;
            }
        }

        class Symbol {
            constructor(name, symbolType, scope, valueType = null) {
                this.name = name;
                this.symbolType = symbolType;
                this.scope = scope;
                this.valueType = valueType;
                this.isInitialized = false;
            }
        }

        class SymbolTable {
            constructor() {
                this.scopes = [{}];
                this.currentScope = 0;
            }

            enterScope() {
                this.currentScope++;
                if (this.scopes.length <= this.currentScope) {
                    this.scopes.push({});
                }
            }

            exitScope() {
                if (this.currentScope > 0) {
                    this.currentScope--;
                }
            }

            declare(name, symbolType, valueType = null) {
                if (name in this.scopes[this.currentScope]) {
                    return false;
                }

                this.scopes[this.currentScope][name] = new Symbol(name, symbolType, this.currentScope, valueType);
                return true;
            }

            lookup(name) {
                for (let i = this.currentScope; i >= 0; i--) {
                    if (name in this.scopes[i]) {
                        return this.scopes[i][name];
                    }
                }
                return null;
            }

            updateInitialization(name) {
                const symbol = this.lookup(name);
                if (symbol) {
                    symbol.isInitialized = true;
                }
            }
        }

        class SemanticAnalyzer {
            constructor(ast) {
                this.ast = ast;
                this.symbolTable = new SymbolTable();
                this.errors = [];
                this.warnings = [];
            }

            analyze() {
                const semanticTree = this.annotateNode(this.ast);
                return { semanticTree, errors: this.errors, warnings: this.warnings };
            }

            annotateNode(node) {
                if (node.nodeType === "Program") {
                    const annotated = new ASTNode("Program", null, [], { scope: 0 });
                    for (const child of node.children) {
                        annotated.children.push(this.annotateNode(child));
                    }
                    return annotated;
                } else if (node.nodeType === "Declaration") {
                    return this.analyzeDeclaration(node);
                } else if (node.nodeType === "FunctionDeclaration") {
                    return this.analyzeFunctionDeclaration(node);
                } else if (node.nodeType === "Assignment") {
                    return this.analyzeAssignment(node);
                } else if (node.nodeType === "BinaryOp") {
                    return this.analyzeBinaryOp(node);
                } else if (node.nodeType === "UnaryOp") {
                    return this.analyzeUnaryOp(node);
                } else if (node.nodeType === "Identifier") {
                    return this.analyzeIdentifier(node);
                } else if (node.nodeType === "Literal") {
                    return this.analyzeLiteral(node);
                } else if (node.nodeType === "FunctionCall") {
                    return this.analyzeFunctionCall(node);
                } else if (node.nodeType === "IfStatement") {
                    return this.analyzeIfStatement(node);
                } else if (node.nodeType === "WhileStatement") {
                    return this.analyzeWhileStatement(node);
                } else if (node.nodeType === "ForStatement") {
                    return this.analyzeForStatement(node);
                } else if (node.nodeType === "ReturnStatement") {
                    return this.analyzeReturnStatement(node);
                } else if (node.nodeType === "Block") {
                    return this.analyzeBlock(node);
                } else {
                    const annotated = new ASTNode(node.nodeType, node.value, [], { ...node.attributes });
                    for (const child of node.children) {
                        annotated.children.push(this.annotateNode(child));
                    }
                    return annotated;
                }
            }

            analyzeDeclaration(node) {
                const varType = node.attributes.type;
                const varName = node.attributes.name;

                if (!this.symbolTable.declare(varName, 'variable', varType)) {
                    this.errors.push(`Variable '${varName}' already declared in this scope`);
                }

                const annotated = new ASTNode("Declaration", null, [], {
                    type: varType,
                    name: varName,
                    scope: this.symbolTable.currentScope,
                    semantic_type: varType
                });

                if (node.children.length > 0) {
                    const initExpr = this.annotateNode(node.children[0]);
                    annotated.children.push(initExpr);

                    const exprType = initExpr.attributes.semantic_type;
                    if (exprType && !this.typesCompatible(varType, exprType)) {
                        this.errors.push(`Type mismatch: cannot assign ${exprType} to ${varType}`);
                    }

                    this.symbolTable.updateInitialization(varName);
                } else {
                    this.warnings.push(`Variable '${varName}' declared but not initialized`);
                }

                return annotated;
            }

            analyzeFunctionDeclaration(node) {
                const funcType = node.attributes.type;
                const funcName = node.attributes.name;
                const params = node.attributes.params || [];

                if (!this.symbolTable.declare(funcName, 'function', funcType)) {
                    this.errors.push(`Function '${funcName}' already declared`);
                }

                this.symbolTable.enterScope();

                for (const param of params) {
                    this.symbolTable.declare(param.name, 'parameter', param.type);
                }

                const annotated = new ASTNode("FunctionDeclaration", null, [], {
                    type: funcType,
                    name: funcName,
                    params: params,
                    scope: this.symbolTable.currentScope - 1,
                    semantic_type: funcType
                });

                for (const child of node.children) {
                    annotated.children.push(this.annotateNode(child));
                }

                this.symbolTable.exitScope();

                return annotated;
            }

            analyzeAssignment(node) {
                const left = this.annotateNode(node.children[0]);
                const right = this.annotateNode(node.children[1]);

                const leftType = left.attributes.semantic_type;
                const rightType = right.attributes.semantic_type;

                if (leftType && rightType && !this.typesCompatible(leftType, rightType)) {
                    this.errors.push(`Type mismatch in assignment: ${leftType} = ${rightType}`);
                }

                if (left.nodeType === "Identifier") {
                    this.symbolTable.updateInitialization(left.value);
                }

                return new ASTNode("Assignment", node.value, [left, right], {
                    semantic_type: leftType || 'unknown'
                });
            }

            analyzeBinaryOp(node) {
                const left = this.annotateNode(node.children[0]);
                const right = this.annotateNode(node.children[1]);

                const leftType = left.attributes.semantic_type;
                const rightType = right.attributes.semantic_type;

                let resultType = this.inferBinaryType(node.value, leftType, rightType);

                if (resultType === 'error') {
                    this.errors.push(`Invalid operation ${node.value} between ${leftType} and ${rightType}`);
                    resultType = 'unknown';
                }

                return new ASTNode("BinaryOp", node.value, [left, right], {
                    semantic_type: resultType
                });
            }

            analyzeUnaryOp(node) {
                const operand = this.annotateNode(node.children[0]);
                const operandType = operand.attributes.semantic_type;

                if (node.value === '-' && !['int', 'float'].includes(operandType)) {
                    this.errors.push(`Unary minus requires numeric type, got ${operandType}`);
                }

                if (node.value === '!' && !['int', 'bool'].includes(operandType)) {
                    this.errors.push(`Logical NOT requires boolean type, got ${operandType}`);
                }

                return new ASTNode("UnaryOp", node.value, [operand], {
                    semantic_type: operandType || 'unknown'
                });
            }

            analyzeIdentifier(node) {
                const symbol = this.symbolTable.lookup(node.value);

                if (!symbol) {
                    this.errors.push(`Undefined identifier '${node.value}'`);
                    return new ASTNode("Identifier", node.value, [], { semantic_type: 'unknown' });
                }

                if (!symbol.isInitialized) {
                    this.warnings.push(`Variable '${node.value}' used before initialization`);
                }

                return new ASTNode("Identifier", node.value, [], {
                    semantic_type: symbol.valueType || 'unknown',
                    scope: symbol.scope,
                    symbol_type: symbol.symbolType
                });
            }

            analyzeLiteral(node) {
                let litType = node.attributes.type || 'unknown';

                if (litType === 'number') {
                    const semanticType = node.value.includes('.') ? 'float' : 'int';
                    return new ASTNode("Literal", node.value, [], {
                        type: litType,
                        semantic_type: semanticType
                    });
                }

                return new ASTNode("Literal", node.value, [], {
                    type: litType,
                    semantic_type: litType
                });
            }

            analyzeFunctionCall(node) {
                const funcName = node.value;
                const symbol = this.symbolTable.lookup(funcName);

                if (!symbol) {
                    this.errors.push(`Undefined function '${funcName}'`);
                    return new ASTNode("FunctionCall", funcName, [], { semantic_type: 'unknown' });
                }

                if (symbol.symbolType !== 'function') {
                    this.errors.push(`'${funcName}' is not a function`);
                }

                const annotated = new ASTNode("FunctionCall", funcName, [], {
                    semantic_type: symbol.valueType || 'unknown',
                    scope: symbol.scope
                });

                for (const child of node.children) {
                    annotated.children.push(this.annotateNode(child));
                }

                return annotated;
            }

            analyzeIfStatement(node) {
                const condition = this.annotateNode(node.children[0]);

                const annotated = new ASTNode("IfStatement", null, [], { scope: this.symbolTable.currentScope });
                annotated.children.push(condition);

                this.symbolTable.enterScope();
                const thenBlock = this.annotateNode(node.children[1]);
                annotated.children.push(thenBlock);
                this.symbolTable.exitScope();

                if (node.children.length > 2) {
                    this.symbolTable.enterScope();
                    const elseBlock = this.annotateNode(node.children[2]);
                    annotated.children.push(elseBlock);
                    this.symbolTable.exitScope();
                }

                return annotated;
            }

            analyzeWhileStatement(node) {
                const condition = this.annotateNode(node.children[0]);

                const annotated = new ASTNode("WhileStatement", null, [], { scope: this.symbolTable.currentScope });
                annotated.children.push(condition);

                this.symbolTable.enterScope();
                const body = this.annotateNode(node.children[1]);
                annotated.children.push(body);
                this.symbolTable.exitScope();

                return annotated;
            }

            analyzeForStatement(node) {
                const annotated = new ASTNode("ForStatement", null, [], { scope: this.symbolTable.currentScope });

                this.symbolTable.enterScope();

                for (const child of node.children) {
                    annotated.children.push(this.annotateNode(child));
                }

                this.symbolTable.exitScope();

                return annotated;
            }

            analyzeReturnStatement(node) {
                const annotated = new ASTNode("ReturnStatement", null, [], { scope: this.symbolTable.currentScope });

                if (node.children.length > 0) {
                    const returnExpr = this.annotateNode(node.children[0]);
                    annotated.children.push(returnExpr);
                    annotated.attributes.semantic_type = returnExpr.attributes.semantic_type;
                } else {
                    annotated.attributes.semantic_type = 'void';
                }

                return annotated;
            }

            analyzeBlock(node) {
                const annotated = new ASTNode("Block", null, [], { scope: this.symbolTable.currentScope });

                for (const child of node.children) {
                    annotated.children.push(this.annotateNode(child));
                }

                return annotated;
            }

            typesCompatible(type1, type2) {
                if (type1 === type2) return true;
                if (['int', 'float'].includes(type1) && ['int', 'float'].includes(type2)) return true;
                return false;
            }

            inferBinaryType(op, leftType, rightType) {
                if (!leftType || !rightType) return 'unknown';

                if (['+', '-', '*', '/'].includes(op)) {
                    if (['int', 'float'].includes(leftType) && ['int', 'float'].includes(rightType)) {
                        return (leftType === 'float' || rightType === 'float') ? 'float' : 'int';
                    }
                    return 'error';
                }

                if (['==', '!=', '<', '>', '<=', '>='].includes(op)) {
                    return this.typesCompatible(leftType, rightType) ? 'int' : 'error';
                }

                if (['&&', '||'].includes(op)) {
                    return 'int';
                }

                return 'unknown';
            }
        }

        function analyzeCode() {
            const sourceCode = document.getElementById('sourceCode').value;

            if (!sourceCode.trim()) {
                alert('Please enter source code to analyze');
                return;
            }

            const lexer = new Lexer(sourceCode);
            const tokens = lexer.tokenize();

            const parser = new Parser(tokens);
            const syntaxTree = parser.parse();

            const semanticAnalyzer = new SemanticAnalyzer(syntaxTree);
            const { semanticTree, errors, warnings } = semanticAnalyzer.analyze();

            const allErrors = [...parser.errors, ...errors];

            displayTokens(tokens);
            displaySyntaxTree(syntaxTree);
            displaySemanticTree(semanticTree);
            displayErrors(allErrors, warnings);
        }

        function displayTokens(tokens) {
            const container = document.getElementById('tokensOutput');
            container.innerHTML = '';

            const nonEofTokens = tokens.filter(t => t.type !== TokenType.EOF);

            if (nonEofTokens.length === 0) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">📝</div><p>No tokens generated.</p></div>';
                return;
            }

            nonEofTokens.forEach(token => {
                const div = document.createElement('div');
                div.className = 'token-item';
                div.innerHTML = `
                    <div class="token-type">${token.type}</div>
                    <div class="token-value">${token.value || '(empty)'}</div>
                    <div class="token-location">L${token.line}:C${token.column}</div>
                `;
                container.appendChild(div);
            });
        }

        function displaySyntaxTree(tree) {
            const container = document.getElementById('syntaxTreeOutput');
            container.innerHTML = '';

            if (!tree) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🌳</div><p>No syntax tree generated.</p></div>';
                return;
            }

            renderTreeNode(tree, container);
        }

        function displaySemanticTree(tree) {
            const container = document.getElementById('semanticTreeOutput');
            container.innerHTML = '';

            if (!tree) {
                container.innerHTML = '<div class="empty-state"><div class="empty-state-icon">🎯</div><p>No semantic tree generated.</p></div>';
                return;
            }

            renderTreeNode(tree, container);
        }

        function renderTreeNode(node, container, depth = 0) {
            const div = document.createElement('div');
            div.className = 'tree-item';

            const hasChildren = node.children && node.children.length > 0;
            const expandIcon = hasChildren ? '▼' : '•';

            let attributesHtml = '';
            if (Object.keys(node.attributes).length > 0) {
                const attrs = Object.entries(node.attributes)
                    .map(([key, val]) => `${key}: ${JSON.stringify(val)}`)
                    .join(', ');
                attributesHtml = `<div class="tree-attributes">${attrs}</div>`;
            }

            div.innerHTML = `
                <div class="tree-item-header">
                    <span class="tree-expand">${expandIcon}</span>
                    <span class="tree-node-type">${node.nodeType}</span>
                    ${node.value !== null && node.value !== undefined ? `<span class="tree-node-value">${node.value}</span>` : ''}
                </div>
                ${attributesHtml}
            `;

            container.appendChild(div);

            if (hasChildren) {
                const childrenContainer = document.createElement('div');
                childrenContainer.className = 'tree-node';

                node.children.forEach(child => {
                    renderTreeNode(child, childrenContainer, depth + 1);
                });

                container.appendChild(childrenContainer);

                div.addEventListener('click', (e) => {
                    e.stopPropagation();
                    const isHidden = childrenContainer.style.display === 'none';
                    childrenContainer.style.display = isHidden ? 'block' : 'none';
                    div.querySelector('.tree-expand').textContent = isHidden ? '▼' : '▶';
                });
            }
        }

        function displayErrors(errors, warnings) {
            const container = document.getElementById('errorsOutput');
            container.innerHTML = '';

            if (errors.length === 0 && warnings.length === 0) {
                const badge = document.createElement('div');
                badge.className = 'status-badge status-success';
                badge.textContent = '✓ No Errors or Warnings';
                container.appendChild(badge);
                return;
            }

            if (errors.length > 0) {
                const badge = document.createElement('div');
                badge.className = 'status-badge status-error';
                badge.textContent = `✗ ${errors.length} Error${errors.length > 1 ? 's' : ''} Found`;
                container.appendChild(badge);

                errors.forEach(error => {
                    const div = document.createElement('div');
                    div.className = 'error-item';
                    div.innerHTML = `
                        <span class="error-icon">✗</span>
                        <span>${error}</span>
                    `;
                    container.appendChild(div);
                });
            }

            if (warnings.length > 0) {
                const badge = document.createElement('div');
                badge.className = 'status-badge status-error';
                badge.style.background = '#fff3cd';
                badge.style.color = '#856404';
                badge.textContent = `⚠ ${warnings.length} Warning${warnings.length > 1 ? 's' : ''}`;
                container.appendChild(badge);

                warnings.forEach(warning => {
                    const div = document.createElement('div');
                    div.className = 'warning-item';
                    div.innerHTML = `
                        <span class="warning-icon">⚠</span>
                        <span>${warning}</span>
                    `;
                    container.appendChild(div);
                });
            }
        }

        function clearAll() {
            document.getElementById('sourceCode').value = '';
            document.getElementById('tokensOutput').innerHTML = '<div class="empty-state"><div class="empty-state-icon">📝</div><p>No tokens generated yet. Run analysis to see results.</p></div>';
            document.getElementById('syntaxTreeOutput').innerHTML = '<div class="empty-state"><div class="empty-state-icon">🌳</div><p>No syntax tree generated yet.</p></div>';
            document.getElementById('semanticTreeOutput').innerHTML = '<div class="empty-state"><div class="empty-state-icon">🎯</div><p>No semantic tree generated yet.</p></div>';
            document.getElementById('errorsOutput').innerHTML = '<div class="empty-state"><div class="empty-state-icon">✅</div><p>No errors or warnings.</p></div>';
        }