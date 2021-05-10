if [ ! -d /home/travis/nest-$NEST_VERSION/lib/python3.6 ] ; then
  export HAS_NEST_CACHE=0
else
  export HAS_NEST_CACHE=1
fi
