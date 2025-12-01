# Compiler Analysis System

This project implements a comprehensive **Compiler Analysis System** capable of performing **lexical, syntax, and semantic analysis** on source code. It provides a structured pipeline to analyze code from raw text input to a fully annotated semantic tree.

## Features

- **Lexical Analysis** : Tokenizes the source code into meaningful elements such as keywords, identifiers, literals, operators, and delimiters.
- **Syntax Analysis** : Generates a syntax tree representing the grammatical structure of the code, validating statements, expressions, and control structures.
- **Semantic Analysis** : Performs type checking, scope resolution, function and variable validation, and reports semantic errors or warnings. Annotates the abstract syntax tree (AST) with semantic information.
- **Error and Warning Reporting** : Collects and reports both syntactic and semantic issues in the code.
- **AST Representation** : Produces a detailed hierarchical representation of the code structure for further analysis or tooling integration.

## Use Case

The system is suitable for projects that require **code validation, error detection, and semantic understanding** , such as educational tools, code editors, or language processing tools. It ensures that programs are not only syntactically correct but also semantically meaningful.

## Output

The system provides:

- A list of **tokens** from lexical analysis.
- A **syntax tree** representing the code’s structure.
- A **semantic tree** with type annotations and scope information.
- A consolidated list of **errors and warnings** .
- A success flag indicating whether the code passed all checks.

This modular design allows easy extension for additional analysis, optimizations, or support for more complex language features.
