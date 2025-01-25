'''
pymmh3 was written by Fredrik Kihlander and enhanced by Swapnil Gusani, and is placed in the public
domain. The authors hereby disclaim copyright to this source code.

pure tensorflow implementation of the murmur3 hash algorithm

https://code.google.com/p/smhasher/wiki/MurmurHash3

This module is written to have the same format as mmh3 python package found here for simple conversions:

https://pypi.python.org/pypi/mmh3/2.3.1
'''

import tensorflow as tf

def hash( key, seed = 0x0 ):
    ''' Implements 32bit murmur3 hash. '''
    tf.debugging.assert_type( key, tf_type=tf.string )

    def fmix( h ):
        h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 16))
        h = tf.math.multiply( h, 0x85ebca6b )
        h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 13))
        h = tf.math.multiply( h, 0xc2b2ae35 )
        h = tf.bitwise.bitwise_xor(h, tf.bitwise.right_shift(h, 16))
        return h

    length = int(tf.strings.length( key ).numpy())
    nblocks = length // 4

    h1 = tf.constant( seed, tf.uint32 )

    c1 = tf.constant(0xcc9e2d51, tf.uint32)
    c2 = tf.constant(0x1b873593, tf.uint32)

    # body
    if nblocks:
        blocks = tf.cast(
            tf.io.decode_raw(
                key, tf.int32, little_endian=True, fixed_length=nblocks * 4),
            tf.uint32 ).numpy()

        for block in blocks:
            k1 = tf.constant( block, tf.uint32 )

            k1 = tf.math.multiply( c1, k1 )
            k1 = tf.bitwise.bitwise_or( tf.bitwise.left_shift( k1, 15 ),
                                        tf.bitwise.right_shift( k1, 17 ))
            k1 = tf.math.multiply( c2, k1 )

            h1 = tf.bitwise.bitwise_xor( h1, k1 )
            h1 = tf.bitwise.bitwise_or( tf.bitwise.left_shift( h1, 13 ),
                                        tf.bitwise.right_shift( h1, 19 ))
            h1  = tf.math.add( tf.math.multiply( h1, 5 ), 0xe6546b64 )

    # tail
    tail_index = nblocks * 4
    k1 = tf.constant( 0, tf.uint32 )
    tail_size = length & 3
    tail_bytes = tf.cast(
        tf.io.decode_raw(tf.strings.substr( key, tail_index, tail_size ),
                         tf.uint8, little_endian=True).numpy(), tf.uint32)
    if tail_size >= 3:
        k1 = tf.bitwise.bitwise_xor(
            k1, tf.bitwise.left_shift( tail_bytes[ 2 ], 16 ) )
    if tail_size >= 2:
        k1 = tf.bitwise.bitwise_xor(
            k1, tf.bitwise.left_shift( tail_bytes[ 1 ], 8 ) )
    if tail_size >= 1:
        k1 = tf.bitwise.bitwise_xor( k1, tail_bytes[0] )

    if tail_size > 0:
        k1  = tf.math.multiply( k1, c1 )
        k1 = tf.bitwise.bitwise_or( tf.bitwise.left_shift( k1, 15 ),
                                    tf.bitwise.right_shift( k1, 17 ))
        k1  = tf.math.multiply( k1, c2 )
        h1 = tf.bitwise.bitwise_xor( h1, k1 )

    #finalization
    unsigned_val = int(fmix( tf.bitwise.bitwise_xor( h1, length ) ).numpy())
    if unsigned_val & 0x80000000 == 0:
        return unsigned_val
    else:
        return -( (unsigned_val ^ 0xFFFFFFFF) + 1 )


def hash128( key, seed = 0x0, x64arch = True ):
    ''' Implements 128bit murmur3 hash. '''

    def hash128_x64( key, seed ):
        ''' Implements 128bit murmur3 hash for x64. '''

        def fmix( k ):
            k = tf.bitwise.bitwise_xor( k, tf.bitwise.right_shift( k, 33 ))
            k = tf.multiply( k, 0xff51afd7ed558ccd )
            k = tf.bitwise.bitwise_xor( k, tf.bitwise.right_shift( k, 33 ))
            k = tf.multiply( k, 0xc4ceb9fe1a85ec53 )
            k = tf.bitwise.bitwise_xor( k, tf.bitwise.right_shift( k, 33 ))
            return k

        length = int(tf.strings.length( key).numpy())
        nblocks = length // 16

        tf.debugging.assert_type( key, tf_type=tf.string )

        h1 = tf.constant( seed, tf.uint64 )
        h2 = h1

        c1 = tf.constant( 0x87c37b91114253d5, tf.uint64 )
        c2 = tf.constant( 0x4cf5ad432745937f, tf.uint64 )

        #body
        if nblocks:
            blocks = tf.cast(
                tf.io.decode_raw(
                    key, tf.int64, little_endian=True, fixed_length=nblocks * 16),
                    tf.uint64 ).numpy()
            for block_start in range( 0, nblocks * 2, 2 ):
                k1 = blocks[block_start]
                k2 = blocks[1 + block_start]

                k1  = tf.math.multiply( c1, k1 )
                k1  = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( k1, 31 ),
                    tf.bitwise.right_shift( k1, 33 ))
                k1 = tf.math.multiply( c2, k1 )
                h1 = tf.bitwise.bitwise_xor( h1, k1 )

                h1  = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( h1, 27 ),
                    tf.bitwise.right_shift( h1, 37 ))
                h1 = tf.math.add( h1, h2 )
                h1 = tf.math.add( tf.math.multiply( h1, 5 ), 0x52dce729 )

                k2 = tf.math.multiply( c2, k2 )
                k2  = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( k2, 33 ),
                    tf.bitwise.right_shift( k2, 31 ))
                k2 = tf.math.multiply( c1, k2 )
                h2 = tf.bitwise.bitwise_xor( h2, k2 )


                h2  = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( h2, 31 ),
                    tf.bitwise.right_shift( h2, 33 ))
                h2 = tf.math.add( h1, h2 )
                h2 = tf.math.add( tf.math.multiply( h2, 5 ), 0x38495ab5 )

        #tail
        tail_index = nblocks * 16
        k1 = tf.constant(0, tf.uint64)
        k2 = tf.constant(0, tf.uint64)
        tail_size = length & 15
        tail_bytes = tf.cast(
            tf.io.decode_raw( tf.strings.substr( key, tail_index, tail_size ),
                              tf.uint8, little_endian=True ).numpy(), tf.uint64 )

        if tail_size >= 15:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 14 ], 48 ) )
        if tail_size >= 14:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 13 ], 40 ) )
        if tail_size >= 13:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 12 ], 32 ) )
        if tail_size >= 12:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 11 ], 24 ) )
        if tail_size >= 11:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 10 ], 16 ) )
        if tail_size >= 10:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift( tail_bytes[ 9 ], 8 ) )
        if tail_size >=  9:
            k2 = tf.bitwise.bitwise_xor( k2, tail_bytes[ 8 ] )

        if tail_size > 8:
            k2 = tf.math.multiply( k2, c2 )
            k2  = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( k2, 33 ),
                    tf.bitwise.right_shift( k2, 31 ))
            k2 = tf.math.multiply( k2, c1 )
            h2 = tf.bitwise.bitwise_xor( h2, k2 )

        if tail_size >=  8:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 7 ], 56 ) )
        if tail_size >=  7:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 6 ], 48 ) )
        if tail_size >=  6:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 5 ], 40 ) )
        if tail_size >=  5:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 4 ], 32 ) )
        if tail_size >=  4:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 3 ], 24 ) )
        if tail_size >=  3:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 2 ], 16 ) )
        if tail_size >=  2:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift( tail_bytes[ 1 ], 8 ) )
        if tail_size >=  1:
            k1 = tf.bitwise.bitwise_xor( k1, tail_bytes[ 0 ] )

        if tail_size > 0:
            k1 = tf.math.multiply( k1, c1 )
            k1  = tf.bitwise.bitwise_or(
                    tf.bitwise.left_shift( k1, 31 ),
                    tf.bitwise.right_shift( k1, 33 ))
            k1 = tf.math.multiply( k1, c2 )
            h1 = tf.bitwise.bitwise_xor( h1, k1 )

        #finalization
        h1 = tf.bitwise.bitwise_xor( h1, length )
        h2 = tf.bitwise.bitwise_xor( h2, length )

        h1  = tf.math.add( h1, h2 )
        h2  = tf.math.add( h1, h2 )

        h1  = fmix( h1 )
        h2  = fmix( h2 )

        h1  = tf.math.add( h1, h2 )
        h2  = tf.math.add( h1, h2 )

        return ( int(h2.numpy()) << 64 | int(h1.numpy()) )

    def hash128_x86( key, seed ):
        ''' Implements 128bit murmur3 hash for x86. '''

        def fmix( h ):
            h ^= h >> 16
            h  = ( h * 0x85ebca6b ) & 0xFFFFFFFF
            h ^= h >> 13
            h  = ( h * 0xc2b2ae35 ) & 0xFFFFFFFF
            h ^= h >> 16
            return h

        length = int(tf.strings.length( key ).numpy())
        nblocks = length // 16
        tf.debugging.assert_type( key, tf_type=tf.string )

        h1 = tf.constant( seed, tf.uint32 )
        h2 = h1
        h3 = h1
        h4 = h1

        c1 = tf.constant( 0x239b961b, tf.uint32 )
        c2 = tf.constant( 0xab0e9789, tf.uint32 )
        c3 = tf.constant( 0x38b34ae5, tf.uint32 )
        c4 = tf.constant( 0xa1e38b93, tf.uint32 )

        #body
        if nblocks:
            blocks = tf.cast(
                tf.io.decode_raw(
                    key, tf.int32, little_endian=True, fixed_length=nblocks * 16),
                tf.uint32).numpy()
            for block_start in range( 0, nblocks * 16, 16 ):
                block_index = block_start // 16

                k1 = blocks[block_index * 4]
                k2 = blocks[block_index * 4 + 1]
                k3 = blocks[block_index * 4 + 2]
                k4 = blocks[block_index * 4 + 3]

                k1  = tf.math.multiply( c1, k1 )
                k1  = tf.bitwise.bitwise_or(
                      tf.bitwise.left_shift( k1, 15 ),
                      tf.bitwise.right_shift( k1, 17 ))
                k1 = tf.math.multiply( c2, k1 )
                h1 = tf.bitwise.bitwise_xor( h1,  k1 )

                h1  = tf.bitwise.bitwise_or(
                      tf.bitwise.left_shift( h1, 19 ),
                      tf.bitwise.right_shift( h1, 13 ))
                h1 = tf.math.add( h1, h2 )
                h1 = tf.math.add( tf.math.multiply( h1, 5 ), 0x561ccd1b )

                k2 = tf.math.multiply( c2, k2 )
                k2  = tf.bitwise.bitwise_or(
                      tf.bitwise.left_shift( k2, 16 ),
                      tf.bitwise.right_shift( k2, 16 ))
                k2 = tf.math.multiply( c3, k2 )
                h2 = tf.bitwise.bitwise_xor( h2, k2 )

                h2  = tf.bitwise.bitwise_or(
                      tf.bitwise.left_shift( h2, 17 ),
                      tf.bitwise.right_shift( h2, 15 ))
                h2 = tf.math.add( h2, h3 )
                h2 = tf.math.add( tf.math.multiply( h2, 5 ), 0x0bcaa747 )

                k3  = tf.math.multiply( c3, k3 )
                k3  = tf.bitwise.bitwise_or(
                      tf.bitwise.left_shift( k3, 17 ),
                      tf.bitwise.right_shift( k3, 15 ))
                k3 = tf.math.multiply( c4, k3 )
                h3 = tf.bitwise.bitwise_xor( h3, k3 )

                h3  = tf.bitwise.bitwise_or(
                      tf.bitwise.left_shift( h3, 15 ),
                      tf.bitwise.right_shift( h3, 17 ))
                h3 = tf.math.add( h3, h4 )
                h3 = tf.math.add( tf.math.multiply( h3, 5 ), 0x96cd1c35 )

                k4  = tf.math.multiply( c4, k4 )
                k4  = tf.bitwise.bitwise_or(
                      tf.bitwise.left_shift( k4, 18 ),
                      tf.bitwise.right_shift( k4, 14 ))
                k4 = tf.math.multiply( c1, k4 )
                h4 = tf.bitwise.bitwise_xor( h4, k4 )

                h4  = tf.bitwise.bitwise_or(
                      tf.bitwise.left_shift( h4, 13 ),
                      tf.bitwise.right_shift( h4, 19 ))
                h4 = tf.math.add( h1, h4 )
                h4 = tf.math.add( tf.math.multiply( h4, 5 ),  0x32ac3b17 )

        #tail
        tail_index = nblocks * 16
        k1 = tf.constant( 0, tf.uint32 )
        k2 = k1
        k3 = k1
        k4 = k1
        tail_size = length & 15
        tail_bytes = tf.cast(
            tf.io.decode_raw( tf.strings.substr( key, tail_index, tail_size ),
                              tf.uint8, little_endian=True ).numpy(), tf.uint32 )

        if tail_size >= 15:
            k4 = tf.bitwise.bitwise_xor(
                k4, tf.bitwise.left_shift(tail_bytes[ 14 ], 16))
        if tail_size >= 14:
            k4 = tf.bitwise.bitwise_xor(
                k4, tf.bitwise.left_shift(tail_bytes[ 13 ], 8))
        if tail_size >= 13:
            k4 = tf.bitwise.bitwise_xor( k4, tail_bytes[ 12 ])

        if tail_size > 12:
            k4 = tf.math.multiply( k4, c4 )
            k4 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k4, 18 ),
                 tf.bitwise.right_shift( k4, 14 ) )
            k4 = tf.math.multiply( k4, c1 )
            h4 = tf.bitwise.bitwise_xor( h4, k4 )

        if tail_size >= 12:
            k3 = tf.bitwise.bitwise_xor(
                k3, tf.bitwise.left_shift(tail_bytes[ 11 ], 24 ) )
        if tail_size >= 11:
            k3 = tf.bitwise.bitwise_xor(
                k3, tf.bitwise.left_shift(tail_bytes[ 10 ], 16 ) )
        if tail_size >= 10:
            k3 = tf.bitwise.bitwise_xor(
                k3, tf.bitwise.left_shift(tail_bytes[ 9 ], 8 ) )
        if tail_size >=  9:
            k3 = tf.bitwise.bitwise_xor( k3, tail_bytes[ 8 ] )

        if tail_size > 8:
            k3 = tf.math.multiply( k3, c3 )
            k3 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k3, 17 ),
                 tf.bitwise.right_shift( k3, 15 ))
            k3 = tf.math.multiply( k3, c4 )
            h3 = tf.bitwise.bitwise_xor( h3, k3 )

        if tail_size >= 8:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift(tail_bytes[ 7 ], 24 ) )
        if tail_size >= 7:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift(tail_bytes[ 6 ], 16 ) )
        if tail_size >= 6:
            k2 = tf.bitwise.bitwise_xor(
                k2, tf.bitwise.left_shift(tail_bytes[ 5 ], 8 ) )
        if tail_size >= 5:
            k2 = tf.bitwise.bitwise_xor( k2, tail_bytes[ 4 ] )

        if tail_size > 4:
            k2 = tf.math.multiply( k2, c2 )
            k2 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k2, 16 ),
                 tf.bitwise.right_shift( k2, 16 ))
            k2 = tf.math.multiply( k2, c3 )
            h2 = tf.bitwise.bitwise_xor( h2, k2 )

        if tail_size >= 4:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift(tail_bytes[ 3 ], 24 ) )
        if tail_size >= 3:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift(tail_bytes[ 2 ], 16 ) )
        if tail_size >= 2:
            k1 = tf.bitwise.bitwise_xor(
                k1, tf.bitwise.left_shift(tail_bytes[ 1 ], 8 ) )
        if tail_size >= 1:
            k1 = tf.bitwise.bitwise_xor( k1, tail_bytes[ 0 ] )

        if tail_size > 0:
            k1 = tf.math.multiply( k1, c1 )
            k1 = tf.bitwise.bitwise_or(
                 tf.bitwise.left_shift( k1, 15 ),
                 tf.bitwise.right_shift( k1, 17 ))
            k1 = tf.math.multiply( k1, c2 )
            h1 = tf.bitwise.bitwise_xor( h1, k1 )

        #finalization
        h1 = int( h1.numpy() )
        h2 = int( h2.numpy() )
        h3 = int( h3.numpy() )
        h4 = int( h4.numpy() )
        h1 ^= length
        h2 ^= length
        h3 ^= length
        h4 ^= length

        h1 = ( h1 + h2 ) & 0xFFFFFFFF
        h1 = ( h1 + h3 ) & 0xFFFFFFFF
        h1 = ( h1 + h4 ) & 0xFFFFFFFF
        h2 = ( h1 + h2 ) & 0xFFFFFFFF
        h3 = ( h1 + h3 ) & 0xFFFFFFFF
        h4 = ( h1 + h4 ) & 0xFFFFFFFF

        h1 = fmix( h1 )
        h2 = fmix( h2 )
        h3 = fmix( h3 )
        h4 = fmix( h4 )

        h1 = ( h1 + h2 ) & 0xFFFFFFFF
        h1 = ( h1 + h3 ) & 0xFFFFFFFF
        h1 = ( h1 + h4 ) & 0xFFFFFFFF
        h2 = ( h1 + h2 ) & 0xFFFFFFFF
        h3 = ( h1 + h3 ) & 0xFFFFFFFF
        h4 = ( h1 + h4 ) & 0xFFFFFFFF

        return ( h4 << 96 | h3 << 64 | h2 << 32 | h1 )

    if x64arch:
        return hash128_x64( key, seed )
    else:
        return hash128_x86( key, seed )


def hash64( key, seed = 0x0, x64arch = True ):
    ''' Implements 64bit murmur3 hash. Returns a tuple. '''
    tf.debugging.assert_type( key, tf_type=tf.string )

    hash_128 = hash128( key, seed, x64arch )

    unsigned_val1 = hash_128 & 0xFFFFFFFFFFFFFFFF
    if unsigned_val1 & 0x8000000000000000 == 0:
        signed_val1 = unsigned_val1
    else:
        signed_val1 = -( (unsigned_val1 ^ 0xFFFFFFFFFFFFFFFF) + 1 )

    unsigned_val2 = ( hash_128 >> 64 ) & 0xFFFFFFFFFFFFFFFF
    if unsigned_val2 & 0x8000000000000000 == 0:
        signed_val2 = unsigned_val2
    else:
        signed_val2 = -( (unsigned_val2 ^ 0xFFFFFFFFFFFFFFFF) + 1 )

    return ( int( signed_val1 ), int( signed_val2 ) )


def hash_bytes( key, seed = 0x0, x64arch = True ):
    ''' Implements 128bit murmur3 hash. Returns a byte string. '''
    tf.debugging.assert_type( key, tf_type=tf.string )

    hash_128 = hash128( key, seed, x64arch )

    bytestring = ''

    for i in range(0, 16, 1):
        lsbyte = hash_128 & 0xFF
        bytestring = bytestring + str( chr( lsbyte ) )
        hash_128 = hash_128 >> 8

    return bytestring


if __name__ == "__main__":
    import argparse
    import sys
    
    parser = argparse.ArgumentParser( 'pymurmur3', 'pymurmur [options] "string to hash"' )
    parser.add_argument( '--seed', type = int, default = 0 )
    parser.add_argument( 'strings', default = [], nargs='+')
    
    opts = parser.parse_args()
    
    for str_to_hash in opts.strings:
        sys.stdout.write( '"%s" = 0x%08X\n' % ( tf.constant(str_to_hash), hash( str_to_hash ) ) )
