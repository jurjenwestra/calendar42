import abc
import time
import threading
import contextlib
import functools
import Queue as queue
import pdb

import collections
#
# -----------------------------------------------------------------------------
# 
class ICache(object):
  """Interface for caches

  Caches may store in memory, on disk or even in a database.

  Caches are similar to defaultdicts. The difference is that you do not provide
  the cache with a default value, but with a functor and arguments that are
  used if the key is not available. Thus, the (potentially expensive) function
  call is not made unnecessarily.

  This interface also allows for (requires) a parallel implementation.
  """
  __metaclass__ = abc.ABCMeta

  @abc.abstractmethod
  def get_cached(self, key, functor=None, pargs=(), kwargs={}):
    """The getter of interest

    @param key
      They key. Typically, this is a combination of functor, pargs, kwargs.
    @param functor
      This is called if key is not available, with pargs as positional
      arguments and kwargs as keyword arguments. The result is stored again
      in the cache so a subsequent call should not result in this functor
      being called
    @param pargs
      See param functor
    @param kwargs
      See param functor
    """
    raise NotImplementedError

  @abc.abstractmethod
  def get_cached_multi(self, key, functor=None, pargs=(), kwargs={}):
    """Get multiple values

    This method exists so it gives implementations the opportunity to implement
    parallel execution.

    The parameters are like @get_cached(), but parallel lists.
    """
    raise NotImplementedError

  @abc.abstractmethod
  def __contains__(self, key):
    raise NotImplementedError

  @abc.abstractmethod
  def __len__(self):
    raise NotImplementedError

  @abc.abstractmethod
  def clear(self):
    """Empty the cache"""
    raise NotImplementedError

  
#
# -----------------------------------------------------------------------------
#
# Helpers
#

@contextlib.contextmanager
def ensure_released(lock):
  """Temporarily unlock a Lock"""
  wasLocked = lock.locked()
  if wasLocked:
    lock.release()
  try:
    yield
  finally:
    if wasLocked:
      lock.acquire()

  
class GetMultiMixin(object):
  """Base class for mixins that provide get_cached_multi()

  This class provides argument checking+cleanup.
  """
  @staticmethod
  def _cleanup_args(key, functor, pargs, kwargs):
    """Make sure the arguments are parallel lists"""
    assert isinstance(key, (list, tuple))
    assert isinstance(pargs, (tuple, list))
    n = len(key)
    
    if not isinstance(functor, (tuple, list)):
      functor = [functor] * n
      
    if len(pargs) == 0:
      pargs = [()] * n
    else:
      if not all([isinstance(x, (tuple, list)) for x in pargs]):
        pargs = [pargs]*n
      elif not len(set([len(x) for x in pargs]))==1: # all same length
        pargs = [pargs]*n
    
    if not isinstance(kwargs, (tuple, list)):
      kwargs = [kwargs] * n

    assert len(functor) == n and len(pargs) == n and len(kwargs) == n
    return key, functor, pargs, kwargs

  @abc.abstractmethod
  def get_cached_multi(self, key, functor=None, pargs=(), kwargs={}):
    raise NotImplementedError


class GetMultiSerialMixin(GetMultiMixin):
  """Provides a serial implementation of get_cached_multi()"""
  def get_cached_multi(self, key, functor=None, pargs=(), kwargs={}):
    key, functor, pargs, kwargs = self._cleanup_args(key, functor,
                                                     pargs, kwargs)
    return [self.get_cached(k,f,pa,kw)
            for k, f, pa,kw in zip(key, functor, pargs, kwargs)]
    

class GetMultiParallelMixin(GetMultiMixin):
  """Provides a parallel implementation of get_cached_multi()

  If keys are not available in the cache, their result is calculated on a
  separate thread. This is a blocking call: the threads are joined before
  we return.
  """
  def get_cached_multi(self, key, functor=None, pargs=(), kwargs={}):
    key, functor, pargs, kwargs = self._cleanup_args(key, functor,
                                                     pargs, kwargs)

    result = [None for k in key]
    calcParallel = []
    for i, (k,f,pa,kw) in enumerate(zip(key, functor, pargs, kwargs)):
      try:
        res = self.get_cached(k,None,pa,kw) # None functor!
        result[i] = res
      except KeyError:
        calcParallel += [i]
    q = queue.Queue()
    threads = [threading.Thread(
        target=self.thread_worker,
        args=(i,q,key[i],functor[i],pargs[i],kwargs[i]))
               for i in calcParallel]
    for th in threads:
      th.start()
    for th in threads:
      th.join()

    for ip in calcParallel:
      i,res = q.get()
      result[i] = res
      
    return result

  def thread_worker(self, i, q, k,f,pa,kw):
    """The function that runs on a worker thread"""
    result = self.get_cached(k,f,pa,kw)
    q.put((i,result))
    return


class CachedDecoratorMixin(object):
  """Provides the cached(...) decorator

  cached simply forwards its arguments to the ctor of the class. Use like this:

  # Implementation of cache
  class MyCache(CachedDecoratorMixin):
    def __init__(self, a,b, xyz):
      ...

  # Users of the cache
  @MyCache.cached(1,2, xyz="xyz")
  def doit(ab, cd):
    ...

  This implementation hides actual cache creation. This implies it is not
  possible to share or manipulate it.
  """
  @classmethod
  def cached(cls, *pargs, **kwargs):
    cache = cls(*pargs, **kwargs)
    def decorator(obj):
      @functools.wraps(obj)
      def wrap(*wpargs, **wkwargs):
        key = (wpargs,
               tuple(sorted(wkwargs.items()))) # dicts are not hashable
        return cache.get_cached(key, obj, wpargs, wkwargs)
      return wrap
    return decorator



def cached(cache):
  """Cache function call based on a previously created cache"""
  def decorator(obj):
    @functools.wraps(obj)
    def wrap(*wpargs, **wkwargs):
      key = (obj, # obj is part of the key!
             wpargs,
             tuple(sorted(wkwargs.items()))) # dicts are not hashable
      return cache.get_cached(key, obj, wpargs, wkwargs)
    return wrap
  return decorator
    


#
# -----------------------------------------------------------------------------
# 
class Stub(object):
  """The cached result + metadata

  This class only requires expireTime as metadata but cache implementations may
  store more if needed.
  """
  def __init__(self, result, expireTime):
    self.result = result
    self.expireTime = expireTime
    return
  
  def is_still_valid(self):
    return self.expireTime > time.time()
  
  def has_expired(self):
    return self.expireTime <= time.time()

  def __repr__(self):
    return "<%s %s exp in %ss at %i>" % (self.__class__.__name__,
                                         self.result,
                                         self.expireTime-time.time(),
                                         id(self))


class ExpiringMemCache(GetMultiSerialMixin, CachedDecoratorMixin, ICache):
  NEVER_EXPIRE_TIME = time.time() + 1000*365*24*60*60. # A millennium from now
  
  def __init__(self, expireSecs, maxNumKeys=None,
               selfCleanup=True, bgCleanup=None):
    self.expireSecs = expireSecs
    self.maxNumKeys = maxNumKeys
    self.selfCleanup = selfCleanup
    self.bgCleanup = bgCleanup
    
    self.lock = threading.Lock()
    self.data = collections.OrderedDict()
    if bgCleanup:
      bgCleanup.register_method(self, "cleanup")
    return
  
  #
  # ICache interface
  #
  def get_cached(self, key, functor=None, pargs=(), kwargs={}):
    with self.lock:
      if self.selfCleanup:
        self._cleanup_maxNumKeys_unsafe()
        self._cleanup_expired_unsafe()
      try:
        stub = self.data[key]
        # key is in data, but it may have expired or being calculated
        if hasattr(stub, "calculationCompleted"):
          with ensure_released(self.lock):
            calculationComplete = stub.calculationCompleted.wait()
        elif stub.has_expired():
          del self.data[key] # Need to remove for order-integrity
          raise KeyError(key)
        return stub.result
      except KeyError:
        if not functor:
          raise
        if self.selfCleanup:
          self._cleanup_maxNumKeys_unsafe(True)
        stub = Stub(None, self.NEVER_EXPIRE_TIME)
        stub.calculationCompleted = threading.Event()
        self.data[key] = stub
        with ensure_released(self.lock):
          # Give others a chance to access data
          res = functor(*pargs, **kwargs)
        stub.result = res
        stub.expireTime = time.time() + self.expireSecs
        stub.calculationCompleted.set()
        delattr(stub, "calculationCompleted")
        return res
        
  # Mixed in / inherited get_cached_multi()

  def __contains__(self, key):
    return key in self.data # No need to lock

  def __len__(self):
    return len(self.data)

  def clear(self):
    with self.lock:
      return self.data.clear()
  #
  # implementation
  #
  def cleanup(self):
    with self.lock:
      self._cleanup_expired_unsafe()
      self._cleanup_maxNumKeys_unsafe()
    return
  
  def _cleanup_expired_unsafe(self):
    try:
      k,v = self.data.items()[0]
      while v.has_expired():
        self.data.popitem(last=False)
        k,v = self.data.items()[0]
    except IndexError:
      pass # dict is empty -- that's ok
    return

  def _cleanup_maxNumKeys_unsafe(self, oneExtra=False):
    if self.maxNumKeys:
      mnk = self.maxNumKeys-1 if oneExtra else self.maxNumKeys
      while len(self.data) > mnk:
        self.data.popitem(last=False)
    return

#
# -----------------------------------------------------------------------------
# 
class ExpiringParallelMemCache(GetMultiParallelMixin, ExpiringMemCache):
  pass
