# CUDA CubeAttack framework #
This is an implementation of a CUDA CubeAttack framework. The code is
copied from my old Master's thesis svn repo (so certain things might
be missing -- if you find that this is the case, please let me know)
at the request of a few students working on a similar topic.
See the [thesis](http://www.deian.net/pubs/stefan:2011:analysis.pdf)
for high-level details of the implementation.

The implementation includes an CUDA implementations of the Xorshift
random number generator, and the Trivium and MICKEY strem ciphers.
See the [gSTREM](https://github.com/deian/gSTREAM) repo for
implementation of other eSTREAM ciphers.
