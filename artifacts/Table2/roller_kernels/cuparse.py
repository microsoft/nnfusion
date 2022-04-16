# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Basic lexer and parser needed
pip install ply
"""

from __future__ import generators
import re
import os.path
import time
import copy
import sys
import ply.yacc as yacc
import ply.lex as lex

# Python 3 compatibility
if sys.version_info.major < 3:
    STRING_TYPES = (str, unicode)
else:
    STRING_TYPES = str
    xrange = range

# tokens for lexing
reserved = {
    # keywords
    'if': 'IF',
    'else': 'ELSE',
    'for': 'FOR',
    'void': 'VOID',
    # data types
    'bool': 'TYPE',
    'char': 'TYPE',
    'double': 'TYPE',
    'float': 'TYPE',
    'int': 'TYPE',
    # qualifiers
    'const': 'QUALIFIER',
    '__volatile__': 'QUALIFIER',
    '__restrict__': 'QUALIFIER',
    # CUDA
    '__shared__': 'SHARED',
    '__syncthreads': 'SYNC',
    '__global__': 'GLOBAL'
}

tokens = ['ID', 'INTEGER', 'FLOAT', 'STRING', 'CHAR', 'WS', 'COMMENT1', 'COMMENT2', 'POUND', 'DPOUND'] + list(
    set(reserved.values()))

literals = "+-*/%|&~^<>=!?()[]{}.,;:\\\'\""


# Whitespace
def t_WS(t):
    r'\s+'
    t.lexer.lineno += t.value.count("\n")
    return t


t_POUND = r'\#'
t_DPOUND = r'\#\#'


# Identifier
def t_ID(t):
    r'[A-Za-z_][\w_]*'
    t.type = reserved.get(t.value, 'ID')
    return t


def t_INTEGER(t):
    r'(((((0x)|(0X))[0-9a-fA-F]+)|(\d+))([uU][lL]|[lL][uU]|[uU]|[lL])?)'
    # t.value = int(t.value)
    return t


# String literal
def t_STRING(t):
    r'\"([^\\\n]|(\\(.|\n)))*?\"'
    t.lexer.lineno += t.value.count("\n")
    return t


# Character constant 'c' or L'c'
def t_CHAR(t):
    r'(L)?\'([^\\\n]|(\\(.|\n)))*?\''
    t.lexer.lineno += t.value.count("\n")
    return t


# Comment
def t_COMMENT1(t):
    r'(/\*(.|\n)*?\*/)'
    ncr = t.value.count("\n")
    t.lexer.lineno += ncr
    # replace with one space or a number of '\n'
    t.type = 'WS'
    t.value = '\n' * ncr if ncr else ' '
    return t


# Line comment
def t_COMMENT2(t):
    r'(//.*?(\n|$))'
    # replace with '/n'
    t.type = 'WS'
    t.value = '\n'
    return t


# error handling
def t_error(t):
    t.type = t.value[0]
    t.value = t.value[0]
    t.lexer.skip(1)
    print("error token encountered", t)
    return t


# rules for parsing shared memory and sync thread

shared_memory = {"symbol": [], "dtype": [], "size": []}
arguments = {"symbol": [], "dtype": []}
signature = []


def p_start(p):
    '''start : signature 
             | shared
    '''


def p_signature(p):
    'signature : GLOBAL VOID ID \'(\' parameters \')\''
    signature.append(p[3])


def p_parameters(p):
    '''parameters : parameter
                  | parameters \',\' parameter
    '''


def p_parameter(p):
    '''parameter : type ID
                 | type QUALIFIER ID
                 | QUALIFIER type ID
    '''
    if (str(p.slice[1]) == "type"):
        arguments["dtype"].append(p[1])
    else:
        arguments["dtype"].append(p[2])
    arguments["symbol"].append(p[len(p) - 1])


def p_type(p):
    '''type : TYPE 
            | TYPE \'*\'
    '''
    # not all cases are covered like: const type * const
    p[0] = ''.join(p[1:])


def p_shared(p):
    'shared : SHARED TYPE ID \'[\' INTEGER \']\' \';\' '
    shared_memory["symbol"].append(p[3])
    shared_memory["dtype"].append(p[2])
    shared_memory["size"].append(int(p[5]))


def p_error(p):
    if p:
        # Just discard the token and tell the parser it's okay.
        parser.errok()
    else:
        print("Syntax error at EOF")


lexer = lex.lex()
parser = yacc.yacc()


def clear_global():
    shared_memory["symbol"].clear()
    shared_memory["dtype"].clear()
    shared_memory["size"].clear()
    arguments["symbol"].clear()
    arguments["dtype"].clear()
    signature.clear()


re_func = re.compile(
    r'(.*__global__\s+void\s+[A-Za-z_]\w*\s*\([^{]*\))\s*({.+\Z)', re.DOTALL)
re_sharedmem = re.compile(
    r'__shared__\s+[A-Za-z_]\w*\s+[A-Za-z_]\w*\s*\[\s*\d+\s*\]\s*;')
re_syncthread = re.compile(r'__syncthreads\s*\(\s*\)\s*;')
print_sync = r'''
  if (blockIdx.x == 0 && blockIdx.y == 0&& blockIdx.z == 0&& threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0)
  {
    printf("Amount of syncthreads logged: %d\n", SYNC_COUNT);
  }
'''


def parse(code, parameters):
    clear_global()

    func_sig, func_body = re_func.match(code).groups()
    new_code = sync_code = func_body

    parser.parse(func_sig)

    for (i, dtype) in enumerate(parameters["dtype"]):
        assert dtype == arguments["dtype"][i]
        if parameters["symbol"][i] != arguments["symbol"][i]:
            new_code = "{} {} = {};\n".format(
                dtype, arguments["symbol"][i], parameters["symbol"][i]) + new_code

    for m in re_sharedmem.finditer(code):
        new_code = new_code.replace(m.group(0), "")
        parser.parse(m.group(0))

    for m in re_syncthread.finditer(code):
        sync_code = sync_code.replace(m.group(0), "__LOGSYNC()\n")
    sync_code = "#define __LOGSYNC() {__syncthreads(); SYNC_COUNT++;}\n" + \
        func_sig + "{\n" + "int SYNC_COUNT = 0;\n" + \
        sync_code + print_sync + "}"

    return func_body, shared_memory, new_code, sync_code, signature[-1]
