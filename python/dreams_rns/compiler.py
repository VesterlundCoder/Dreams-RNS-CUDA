"""
CMF Compiler: Converts symbolic CMF expressions to GPU bytecode.

This module takes CMF definitions from ramanujantools and compiles them
into a bytecode format that can be efficiently executed on GPU.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
import struct
import numpy as np

class Opcode(IntEnum):
    """Bytecode opcodes for GPU evaluator."""
    NOP = 0
    LOAD_X = 1      # Load axis value
    LOAD_C = 2      # Load constant
    LOAD_N = 3      # Load step number
    ADD = 4
    SUB = 5
    MUL = 6
    NEG = 7
    POW2 = 8
    POW3 = 9
    INV = 10
    STORE = 11
    END = 12


@dataclass
class Instruction:
    """Single bytecode instruction."""
    op: Opcode
    arg0: int = 0
    arg1: int = 0
    arg2: int = 0
    
    def pack(self) -> bytes:
        """Pack instruction to binary format."""
        return struct.pack('IIII', int(self.op), self.arg0, self.arg1, self.arg2)


@dataclass 
class CmfProgram:
    """Compiled CMF program ready for GPU execution."""
    m: int                              # Matrix dimension
    dim: int                            # Number of axes
    opcodes: List[Instruction] = field(default_factory=list)
    constants: List[int] = field(default_factory=list)
    directions: List[int] = field(default_factory=list)
    
    def serialize(self) -> bytes:
        """Serialize program to binary format."""
        data = struct.pack('IIII', self.m, self.dim, len(self.opcodes), len(self.constants))
        for instr in self.opcodes:
            data += instr.pack()
        for c in self.constants:
            data += struct.pack('q', c)  # 64-bit signed
        for d in self.directions:
            data += struct.pack('i', d)
        return data
    
    @classmethod
    def deserialize(cls, data: bytes) -> 'CmfProgram':
        """Deserialize program from binary format."""
        offset = 0
        m, dim, n_ops, n_const = struct.unpack_from('IIII', data, offset)
        offset += 16
        
        opcodes = []
        for _ in range(n_ops):
            op, a0, a1, a2 = struct.unpack_from('IIII', data, offset)
            opcodes.append(Instruction(Opcode(op), a0, a1, a2))
            offset += 16
        
        constants = []
        for _ in range(n_const):
            c, = struct.unpack_from('q', data, offset)
            constants.append(c)
            offset += 8
        
        directions = []
        for _ in range(dim):
            d, = struct.unpack_from('i', data, offset)
            directions.append(d)
            offset += 4
        
        return cls(m=m, dim=dim, opcodes=opcodes, constants=constants, directions=directions)


class CmfCompiler:
    """
    Compiles CMF symbolic expressions to GPU bytecode.
    
    Usage:
        compiler = CmfCompiler(m=4, dim=2)
        # Add matrix entries as expressions
        compiler.compile_entry(0, 0, "x0 + 1")
        compiler.compile_entry(0, 1, "x1 * 2")
        # ... etc
        program = compiler.build()
    """
    
    MAX_REGS = 64
    
    def __init__(self, m: int, dim: int, directions: Optional[List[int]] = None):
        self.m = m
        self.dim = dim
        self.directions = directions or [1] * dim
        self.opcodes: List[Instruction] = []
        self.constants: Dict[int, int] = {}  # value -> index
        self.next_reg = 0
        self.entry_regs: Dict[Tuple[int, int], int] = {}  # (row, col) -> reg
    
    def _alloc_reg(self) -> int:
        """Allocate a new register."""
        reg = self.next_reg
        self.next_reg += 1
        if self.next_reg >= self.MAX_REGS:
            raise RuntimeError("Out of registers")
        return reg
    
    def _add_const(self, value: int) -> int:
        """Add a constant and return its index."""
        if value not in self.constants:
            self.constants[value] = len(self.constants)
        return self.constants[value]
    
    def _emit(self, op: Opcode, arg0: int = 0, arg1: int = 0, arg2: int = 0):
        """Emit an instruction."""
        self.opcodes.append(Instruction(op, arg0, arg1, arg2))
    
    def load_axis(self, axis: int) -> int:
        """Load axis value into a register."""
        reg = self._alloc_reg()
        self._emit(Opcode.LOAD_X, axis, reg)
        return reg
    
    def load_const(self, value: int) -> int:
        """Load constant into a register."""
        idx = self._add_const(value)
        reg = self._alloc_reg()
        self._emit(Opcode.LOAD_C, idx, reg)
        return reg
    
    def load_n(self) -> int:
        """Load step number into a register."""
        reg = self._alloc_reg()
        self._emit(Opcode.LOAD_N, reg)
        return reg
    
    def add(self, r1: int, r2: int) -> int:
        """Add two registers."""
        dest = self._alloc_reg()
        self._emit(Opcode.ADD, r1, r2, dest)
        return dest
    
    def sub(self, r1: int, r2: int) -> int:
        """Subtract two registers."""
        dest = self._alloc_reg()
        self._emit(Opcode.SUB, r1, r2, dest)
        return dest
    
    def mul(self, r1: int, r2: int) -> int:
        """Multiply two registers."""
        dest = self._alloc_reg()
        self._emit(Opcode.MUL, r1, r2, dest)
        return dest
    
    def neg(self, r: int) -> int:
        """Negate a register."""
        dest = self._alloc_reg()
        self._emit(Opcode.NEG, r, dest)
        return dest
    
    def pow2(self, r: int) -> int:
        """Square a register."""
        dest = self._alloc_reg()
        self._emit(Opcode.POW2, r, dest)
        return dest
    
    def pow3(self, r: int) -> int:
        """Cube a register."""
        dest = self._alloc_reg()
        self._emit(Opcode.POW3, r, dest)
        return dest
    
    def inv(self, r: int) -> int:
        """Modular inverse of a register."""
        dest = self._alloc_reg()
        self._emit(Opcode.INV, r, dest)
        return dest
    
    def store(self, r: int, row: int, col: int):
        """Store register to matrix entry."""
        self._emit(Opcode.STORE, r, row, col)
        self.entry_regs[(row, col)] = r
    
    def compile_sympy_expr(self, expr, axis_symbols: Dict[str, int]) -> int:
        """
        Compile a SymPy expression to bytecode.
        
        Args:
            expr: SymPy expression
            axis_symbols: mapping from symbol names to axis indices
        
        Returns:
            Register containing the result
        """
        import sympy as sp
        
        if isinstance(expr, (int, sp.Integer)):
            return self.load_const(int(expr))
        
        if isinstance(expr, sp.Symbol):
            name = str(expr)
            if name == 'n':
                return self.load_n()
            if name in axis_symbols:
                return self.load_axis(axis_symbols[name])
            raise ValueError(f"Unknown symbol: {name}")
        
        if isinstance(expr, sp.Add):
            args = expr.args
            result = self.compile_sympy_expr(args[0], axis_symbols)
            for arg in args[1:]:
                r = self.compile_sympy_expr(arg, axis_symbols)
                result = self.add(result, r)
            return result
        
        if isinstance(expr, sp.Mul):
            args = expr.args
            # Handle negative: -1 * x
            if args[0] == -1:
                r = self.compile_sympy_expr(sp.Mul(*args[1:]), axis_symbols)
                return self.neg(r)
            result = self.compile_sympy_expr(args[0], axis_symbols)
            for arg in args[1:]:
                r = self.compile_sympy_expr(arg, axis_symbols)
                result = self.mul(result, r)
            return result
        
        if isinstance(expr, sp.Pow):
            base, exp = expr.args
            if exp == 2:
                r = self.compile_sympy_expr(base, axis_symbols)
                return self.pow2(r)
            if exp == 3:
                r = self.compile_sympy_expr(base, axis_symbols)
                return self.pow3(r)
            if exp == -1:
                r = self.compile_sympy_expr(base, axis_symbols)
                return self.inv(r)
            # General power: expand to multiplications
            if isinstance(exp, (int, sp.Integer)) and int(exp) > 0:
                r = self.compile_sympy_expr(base, axis_symbols)
                result = r
                for _ in range(int(exp) - 1):
                    result = self.mul(result, r)
                return result
            raise ValueError(f"Unsupported power: {expr}")
        
        if isinstance(expr, sp.Rational):
            num = self.load_const(int(expr.p))
            den = self.load_const(int(expr.q))
            den_inv = self.inv(den)
            return self.mul(num, den_inv)
        
        raise ValueError(f"Unsupported expression type: {type(expr)}")
    
    def compile_matrix_entry(self, row: int, col: int, expr, 
                             axis_symbols: Dict[str, int]):
        """Compile a matrix entry expression and store it."""
        reg = self.compile_sympy_expr(expr, axis_symbols)
        self.store(reg, row, col)
    
    def build(self) -> CmfProgram:
        """Build the final program."""
        self._emit(Opcode.END)
        
        # Convert constants dict to list
        const_list = [0] * len(self.constants)
        for val, idx in self.constants.items():
            const_list[idx] = val
        
        return CmfProgram(
            m=self.m,
            dim=self.dim,
            opcodes=self.opcodes.copy(),
            constants=const_list,
            directions=self.directions.copy()
        )


def compile_cmf(cmf, axis_names: Optional[List[str]] = None) -> CmfProgram:
    """
    Compile a CMF object from ramanujantools to GPU bytecode.
    
    Args:
        cmf: CMF object from ramanujantools
        axis_names: Optional list of axis symbol names (e.g., ['x', 'y'])
    
    Returns:
        Compiled CmfProgram ready for GPU execution
    """
    try:
        import sympy as sp
        from sympy import Matrix
    except ImportError:
        raise ImportError("SymPy is required for CMF compilation")
    
    # Get matrix from CMF
    if hasattr(cmf, 'M'):
        # ramanujantools CMF object
        M = cmf.M
        m = M.rows
    elif hasattr(cmf, 'matrix'):
        M = cmf.matrix
        m = M.rows
    else:
        raise ValueError("CMF object must have 'M' or 'matrix' attribute")
    
    # Determine axis symbols
    if axis_names is None:
        # Try to extract from CMF
        if hasattr(cmf, 'symbols'):
            axis_names = [str(s) for s in cmf.symbols]
        else:
            # Default: x0, x1, x2, ...
            free_syms = M.free_symbols
            axis_names = sorted([str(s) for s in free_syms if str(s) != 'n'])
    
    dim = len(axis_names)
    axis_symbols = {name: i for i, name in enumerate(axis_names)}
    
    # Get directions if available
    directions = [1] * dim
    if hasattr(cmf, 'directions'):
        directions = list(cmf.directions)
    
    # Compile
    compiler = CmfCompiler(m=m, dim=dim, directions=directions)
    
    for i in range(m):
        for j in range(m):
            entry = M[i, j]
            if entry != 0:  # Skip zero entries
                compiler.compile_matrix_entry(i, j, entry, axis_symbols)
    
    return compiler.build()


def compile_cmf_from_dict(matrix_dict: Dict[Tuple[int, int], str], 
                          m: int, dim: int,
                          axis_names: Optional[List[str]] = None,
                          directions: Optional[List[int]] = None) -> CmfProgram:
    """
    Compile a CMF from a dictionary of matrix entries.
    
    Args:
        matrix_dict: {(row, col): "expression"} dictionary
        m: Matrix dimension
        dim: Number of axes
        axis_names: List of axis symbol names
        directions: Walk directions for each axis
    
    Example:
        program = compile_cmf_from_dict(
            {(0, 0): "x + 1", (0, 1): "y * 2", (1, 0): "1", (1, 1): "x * y"},
            m=2, dim=2, axis_names=['x', 'y']
        )
    """
    import sympy as sp
    
    if axis_names is None:
        axis_names = [f'x{i}' for i in range(dim)]
    
    axis_symbols = {name: i for i, name in enumerate(axis_names)}
    
    # Create SymPy symbols
    sym_dict = {name: sp.Symbol(name) for name in axis_names}
    sym_dict['n'] = sp.Symbol('n')
    
    compiler = CmfCompiler(m=m, dim=dim, directions=directions or [1]*dim)
    
    for (row, col), expr_str in matrix_dict.items():
        expr = sp.sympify(expr_str, locals=sym_dict)
        compiler.compile_matrix_entry(row, col, expr, axis_symbols)
    
    return compiler.build()
