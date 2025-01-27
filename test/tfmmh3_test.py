# pymmh3 was written by Fredrik Kihlander and enhanced by Swapnil Gusani, and is placed in the public
# domain. The authors hereby disclaim copyright to this source code.
# tfmmh3 by Jeffrey Sorensen, and is placed in the public domain.

import os
import sys

import tensorflow as tf

file_dir = os.path.dirname( __file__ )
sys.path.append( os.path.join( file_dir, '..' ) )
import tfmmh3

class Testtfmmh3( tf.test.TestCase ):
    def _load_solutions(self, solution_file, base = 16):
        solution = {}
        with open( os.path.join( file_dir, solution_file ), 'rb' ) as f:
            while True:
                l = f.readline()
                if not l:
                    break
                solution[ l ] = int( f.readline(), base )

        return solution
        
    
    def test_32bit_basic_string( self ):
        solution = self._load_solutions('solution_hash32_seed0.txt', 10)

        with open( os.path.join( file_dir, 'pg1260.txt' ), 'rb' ) as test_file:
            for l in test_file.readlines():
                s = solution[l]
                r = tfmmh3.hash( tf.constant(l) )
                self.assertEqual( s, r, 'different hash for line: "%s"\n0x%08X != 0x%08X' % ( l, s, r ) )

    def test_32bit_custom_seed_string( self ):
        solution = self._load_solutions('solution_hash32_seed1234ABCD.txt', 10)

        with open( os.path.join( file_dir, 'pg1260.txt' ), 'rb' ) as test_file:
            for l in test_file.readlines():
                s = solution[l]
                r = tfmmh3.hash( tf.constant(l), seed = tf.constant( 0x1234ABCD, tf.uint32 ) )
                self.assertEqual( s, r, 'different hash for line: "%s"\n0x%08X != 0x%08X' % ( l, s, r ) )

    def test_128bit_x86_basic_string( self ):
        solution = self._load_solutions('solution_hash128_x86_seed0.txt')

        with open( os.path.join( file_dir, 'pg1260.txt' ), 'rb' ) as test_file:
            for l in test_file.readlines():
                s = solution[l]
                h = tfmmh3.hash128( tf.constant(l) , x64arch = False )
                r = ( int( h.numpy()[ 0 ] ) << 64 ) + int( h.numpy()[ 1 ] )
                self.assertEqual( s, r, 'different hash for line: "%s"\n0x%08X != 0x%08X' % ( l, s, r ) )

    def test_128bit_x86_custom_seed_string( self ):
        solution = self._load_solutions('solution_hash128_x86_seed1234ABCD.txt')

        with open( os.path.join( file_dir, 'pg1260.txt' ), 'rb' ) as test_file:
            for l in test_file.readlines():
                s = solution[l]
                h = tfmmh3.hash128( tf.constant(l), seed = tf.constant( 0x1234ABCD, tf.uint32 ),
                                    x64arch = False )
                r = ( int( h.numpy()[ 0 ] ) << 64 ) + int( h.numpy()[ 1 ] )
                self.assertEqual( s, r, 'different hash for line: "%s"\n0x%08X != 0x%08X' % ( l, s, r ) )

    def test_128bit_x64_basic_string( self ):
        solution = self._load_solutions('solution_hash128_x64_seed0.txt')

        with open( os.path.join( file_dir, 'pg1260.txt' ), 'rb' ) as test_file:
            for l in test_file.readlines():
                s = solution[l]
                h = tfmmh3.hash128( tf.constant(l), x64arch = True )
                r = ( int( h.numpy()[ 0 ] ) << 64 ) + int( h.numpy()[ 1 ] )
                self.assertEqual( s, r, 'different hash for line: "%s"\n0x%08X != 0x%08X' % ( l, s, r ) )

    def test_128bit_x64_custom_seed_string( self ):
        solution = self._load_solutions('solution_hash128_x64_seed1234ABCD.txt')

        with open( os.path.join( file_dir, 'pg1260.txt' ), 'rb' ) as test_file:
            for l in test_file.readlines():
                s = solution[l]
                h = tfmmh3.hash128( tf.constant(l), seed = tf.constant( 0x1234ABCD, tf.uint32 ),
                                    x64arch = True )
                r = ( int( h.numpy()[ 0 ] ) << 64 ) + int( h.numpy()[ 1 ] )
                self.assertEqual( s, r, 'different hash for line: "%s"\n0x%08X != 0x%08X' % ( l, s, r ) )

if __name__ == "__main__":
    tf.test.main()
