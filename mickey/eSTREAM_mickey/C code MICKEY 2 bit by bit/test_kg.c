/* This file contains code to produce test data for a keystream generator */

/* Date started      :- 19/4/05 */
/* Date last altered :- 28/4/05 */

#include <stdio.h>

#include "ecrypt-sync.h"

/* The following routine generates 16 bytes of keystream from key and iv, and
   displays key, iv and keystream */

void perform_test (u8 *key, u8* iv, int iv_length_in_bits)
{
    ECRYPT_ctx ctx;
        /* Keystream generator context */
    u8 keystream[16];
        /* Array to contain generated keystream bytes */
    int i;
        /* Counting variable */

    /* Load key */
    ECRYPT_keysetup (&ctx, key, 80, iv_length_in_bits);
    /* Load IV */
    ECRYPT_ivsetup (&ctx, iv);
    /* Generate keystream */
    ECRYPT_keystream_bytes (&ctx, keystream, 16);

    /* Display the key */
    printf ("Key               =");
    for (i=0; i<10; i++)
        printf (" %02x", key[i]);
    printf ("\n");

    /* Display the IV */
    printf ("IV                =");
    for (i=0; i<(iv_length_in_bits+7)/8; i++)
        printf (" %02x", iv[i]);
    printf ("\n");

    /* Display the derived keytream */
    printf ("Keystream         =");
    for (i=0; i<16; i++)
        printf (" %02x", keystream[i]);
    printf ("\n");

    printf ("\n");

}

/* The following routine repeatedly reloads the keystream generator with key and iv
   formed from keystream from a previous load of the generator.  */

void perform_iterated_test (u8 *key)
{
    ECRYPT_ctx ctx;
        /* Keystream generator context */
    u8 iv[4];
        /* Array to contain iv derived from keystream */
    u8 keystream[16];
        /* Array to contain generated keystream bytes */
    int i;
        /* Counting variable */

    /* Display the key */
    printf ("Iterated test key =");
    for (i=0; i<10; i++)
        printf (" %02x", key[i]);
    printf ("\n");

    /* Load key */
    ECRYPT_keysetup (&ctx, key, 80, 0);
    ECRYPT_ivsetup (&ctx, iv);

    for (i=0; i<1000; i++)
    {
        /* Generate new key and iv from keystream */
        ECRYPT_keystream_bytes (&ctx, key, 10);
        ECRYPT_keystream_bytes (&ctx, iv, 4);

        /* Load new key */
        ECRYPT_keysetup (&ctx, key, 80, 32);

        /* Load new IV */
        ECRYPT_ivsetup (&ctx, iv);
    }

    /* Generate keystream */
    ECRYPT_keystream_bytes (&ctx, keystream, 16);

    /* Display the derived keytream */
    printf ("Final keystream   =");
    for (i=0; i<16; i++)
        printf (" %02x", keystream[i]);
    printf ("\n");

    printf ("\n");

}

/* Main program */

void main()
{
    u8 key_1[10] = {0x12, 0x34, 0x56, 0x78, 0x9a, 0xbc, 0xde, 0xf0, 0x12, 0x34};
    u8 key_2[10] = {0xf1, 0x1a, 0x56, 0x27, 0xce, 0x43, 0xb6, 0x1f, 0x89, 0x12};
    u8 key_3[10] = {0x3b, 0x80, 0xfc, 0x8c, 0x47, 0x5f, 0xc2, 0x70, 0xfa, 0x26};
    u8 key_4[10] = {0x82, 0xac, 0xb3, 0x8c, 0x5d, 0x7a, 0x3c, 0x78, 0xd9, 0x8f};
    u8 iv_1[4]   = {0x21, 0x43, 0x65, 0x87};
    u8 iv_2[10]  = {0x9c, 0x53, 0x2f, 0x8a, 0xc3, 0xea, 0x4b, 0x2e, 0xa0, 0xf5};

    /* Initialise the algorithm */
    ECRYPT_init ();

    /* Generate the test data */
    perform_test (key_1, iv_1, 32);
    perform_test (key_2, iv_2, 80);
    perform_test (key_3, NULL, 0);
    perform_iterated_test (key_4);
}