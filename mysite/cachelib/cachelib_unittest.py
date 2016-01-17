import unittest

import os
import time
import itertools
import threading
import Queue as queue
import weakref
import pdb

import sys
sys.path += [os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))]
import cachelib # Module-under-test

#
# -----------------------------------------------------------------------------
#
# Helper classes
#

class MethodTask(object):
  def __init__(self, instance, methodName):
    self.instance = weakref.ref(instance)
    self.methodName = methodName
    return

  def is_alive(self):
    return not self.instance() is None

  def __call__(self):
    instance = self.instance()
    if instance is None:
      return # We have a dead ref -- simply do nothing
    return  getattr(instance, self.methodName)()


class BackgroundExecuter(object):
  """A dummy background executer

  Normally, background tasks run on their own thread, and run e.g. at specific
  intervals / times.
  """
  def __init__(self):
    self.tasks = []
    return
  
  def register_method(self, instance, methodName):
    self.tasks += [MethodTask(instance, methodName)]
    return

  def execute(self):
    # Remove dead tasks
    self.tasks = [t for t in self.tasks if t.is_alive()]
    for t in self.tasks:
      t()
    return
#
# -----------------------------------------------------------------------------
# 
class TestExpiringMemCache(unittest.TestCase):
  def test_insert_expire_clear(self):
    EXPIRE_SECS = 0.1
    cache = cachelib.ExpiringMemCache(EXPIRE_SECS, None)

    # Do this multiple times to make sure things are working after clear()
    for i in xrange(3):
      f = itertools.count().next
      assert len(cache) == 0
      self.assertEqual(cache.get_cached(0, f), 0)

      time.sleep(0.5*EXPIRE_SECS)
      self.assertEqual(cache.get_cached(1, f), 1)
      self.assertEqual(cache.get_cached(0, f), 0)
      self.assertEqual(cache.get_cached(0, f), 0)
      self.assertEqual(cache.get_cached(1, f), 1)

      time.sleep(0.6*EXPIRE_SECS)
      self.assertEqual(cache.get_cached(0, f), 2)
      self.assertEqual(cache.get_cached(1, f), 1)

      time.sleep(0.6*EXPIRE_SECS)
      self.assertEqual(cache.get_cached(0, f), 2)
      self.assertEqual(cache.get_cached(1, f), 3)

      time.sleep(1.1*EXPIRE_SECS)
      self.assertEqual(cache.get_cached(0, f), 4)
      self.assertEqual(cache.get_cached(1, f), 5)

      self.assertEqual(len(cache), 2)
      self.assertTrue(0 in cache)
      self.assertTrue(1 in cache)
      self.assertFalse("abc" in cache)

      cache.clear()
    return

  def test_insert_expire_args(self):
    def echo(*pargs, **kwargs):
      return pargs, kwargs
    
    EXPIRE_SECS = 0.1
    cache = cachelib.ExpiringMemCache(EXPIRE_SECS, None)
    f = echo
    pargs, kwargs = (0,),dict(a=0)
    self.assertEqual(cache.get_cached(0, f, pargs, kwargs),
                     (pargs, kwargs))

    time.sleep(0.5*EXPIRE_SECS)
    pargs2, kwargs2 = (1,),dict(b=1)
    self.assertEqual(cache.get_cached(1, f, pargs, kwargs),
                     (pargs, kwargs))
    self.assertEqual(cache.get_cached(0, f, pargs2, kwargs2),
                     (pargs, kwargs))
    self.assertEqual(cache.get_cached(0, f, pargs2, kwargs2),
                     (pargs, kwargs))
    self.assertEqual(cache.get_cached(1, f, pargs2, kwargs2),
                     (pargs, kwargs))

    
    time.sleep(0.6*EXPIRE_SECS)
    pargs3, kwargs3 = (3,),dict(c=3)
    self.assertEqual(cache.get_cached(0, f, pargs3, kwargs3),
                     (pargs3, kwargs3))
    self.assertEqual(cache.get_cached(1, f, pargs3, kwargs3),
                     (pargs, kwargs))

    time.sleep(0.6*EXPIRE_SECS)
    pargs4, kwargs4 = (4,),dict(d=4)
    self.assertEqual(cache.get_cached(0, f, pargs4, kwargs4),
                     (pargs3, kwargs3))
    self.assertEqual(cache.get_cached(1, f, pargs4, kwargs4),
                     (pargs4, kwargs4))
    
    time.sleep(1.1*EXPIRE_SECS)
    pargs5, kwargs5 = (5,),dict(e=5)
    self.assertEqual(cache.get_cached(0, f, pargs5, kwargs5),
                     (pargs5, kwargs5))
    self.assertEqual(cache.get_cached(1, f, pargs5, kwargs5),
                     (pargs5, kwargs5))

    self.assertEqual(len(cache), 2)
    self.assertTrue(0 in cache)
    self.assertTrue(1 in cache)
    self.assertFalse("abc" in cache)
    return

  def test_insert_expire_multi(self):
    EXPIRE_SECS = 0.1
    cache = cachelib.ExpiringMemCache(EXPIRE_SECS, None)
    f = itertools.count().next
  
    self.assertEqual(cache.get_cached_multi([0,1], f), [0,1])

    time.sleep(1.1*EXPIRE_SECS)
    self.assertEqual(cache.get_cached(0, f), 2)

    time.sleep(0.6*EXPIRE_SECS)
    self.assertEqual(cache.get_cached_multi([0,1], f), [2,3])

    time.sleep(0.6*EXPIRE_SECS)
    self.assertEqual(cache.get_cached_multi([0,1], f), [4,3])

    self.assertEqual(len(cache), 2)
    self.assertTrue(0 in cache)
    self.assertTrue(1 in cache)
    self.assertFalse("abc" in cache)
    return

  def test_maxNumKeys(self):
    EXPIRE_SECS = 0.1
    maxNumKeys = 2
    cache = cachelib.ExpiringMemCache(EXPIRE_SECS, maxNumKeys)
    f = itertools.count().next

    self.assertEqual(cache.get_cached(0, f), 0)
    self.assertEqual(len(cache), 1)
    self.assertTrue(0 in cache)
    self.assertFalse(1 in cache)
    
    self.assertEqual(cache.get_cached(1, f), 1)
    self.assertEqual(len(cache), 2)
    self.assertTrue(0 in cache)
    self.assertTrue(1 in cache)

    self.assertEqual(cache.get_cached(0, f), 0)
    self.assertEqual(len(cache), 2)
    self.assertTrue(0 in cache)
    self.assertTrue(1 in cache)
    
    # Overflow here
    self.assertEqual(cache.get_cached(2, f), 2)
    self.assertEqual(len(cache), 2)

    self.assertFalse(0 in cache)
    self.assertTrue(1 in cache)
    self.assertTrue(2 in cache)

    time.sleep(1.1*EXPIRE_SECS)
    self.assertEqual(cache.get_cached(0, f), 3)
    self.assertEqual(len(cache), 1)
    

    return

  def test_maxNumKeys_cleanup(self):
    EXPIRE_SECS = 0.1
    maxNumKeys = 2

    cache1 = cachelib.ExpiringMemCache(EXPIRE_SECS, maxNumKeys)
    f1 = itertools.count().next
    cache2 = cachelib.ExpiringMemCache(EXPIRE_SECS, maxNumKeys,
                                       selfCleanup=False)
    f2 = itertools.count().next

    for cache,f,cleaned in [(cache1,f1,True), (cache2,f2,False)]:
      self.assertEqual(cache.get_cached(0, f), 0)
      self.assertEqual(len(cache), 1)
      self.assertTrue(0 in cache)
      self.assertFalse(1 in cache)

      self.assertEqual(cache.get_cached(1, f), 1)
      self.assertEqual(len(cache), 2)
      self.assertTrue(0 in cache)
      self.assertTrue(1 in cache)

      self.assertEqual(cache.get_cached(0, f), 0)
      self.assertEqual(len(cache), 2)
      self.assertTrue(0 in cache)
      self.assertTrue(1 in cache)

      # Overflow here
      self.assertEqual(cache.get_cached(2, f), 2)
      if cleaned:
        self.assertEqual(len(cache), 2)
        self.assertFalse(0 in cache)
      else:
        self.assertEqual(len(cache), 3)
        self.assertTrue(0 in cache)
        
      self.assertTrue(1 in cache)
      self.assertTrue(2 in cache)

      time.sleep(1.1*EXPIRE_SECS)
      self.assertEqual(cache.get_cached(0, f), 3)
      if cleaned:
        self.assertEqual(len(cache), 1)
      else:
        self.assertEqual(len(cache), 3)

    return

  def test_threaded_parallel(self):
    """Test basica parallel access to a cache

    This test uses threads and tries to time certain events. Timing is a bit
    inaccurate, so we use fudge factors. This test may actually fail if the
    fudge factors are not sufficient.
    """
    flock = threading.Lock()
    EXPIRE_TIME = 1.0
    CALC_TIME = 0.2
    T1_START_TIME = 0.1
    
    def f(cnt, key):
      """Takes about 0.2s to calculate"""
      time.sleep(CALC_TIME)
      with flock:
        return count.next()

    def worker(slp, cache, func, cnt, key, q):
      time.sleep(slp)
      res = cache.get_cached(key, func, (cnt,key))
      q.put((key,res))
      return

    cache = cachelib.ExpiringMemCache(expireSecs=EXPIRE_TIME, maxNumKeys=None)
    count = itertools.count()
    q = queue.Queue()

    # 0,0
    t0 = threading.Thread(target=worker, args=(0., cache, f, count,0, q))  
    # 1,1 -- while t0 is busy calculating
    t1 = threading.Thread(target=worker, args=(T1_START_TIME, cache, f, count,1, q))

    threads = [t0,t1]
    for t in threads:
      t.daemon = True

    startTime = time.time()
    for t in threads:
      t.start()
    for t in threads:
      t.join()
    endTime = time.time()


    result = []
    sentinel = object()
    q.put(sentinel)    
    x = q.get()
    while not x is sentinel:
      result += [ x ]
      x = q.get()
    
    self.assertEqual(result, [(0,0), (1,1)])

    # Need fudge factor 0.001?!?!
    self.assertTrue( endTime-startTime >= T1_START_TIME + CALC_TIME - 0.001 ,
                     msg="startTime: %s, endTime: %s, endTime-startTime:%s" % (
                       startTime, endTime, endTime-startTime))
    self.assertEqual( len(cache), 2 )
    # Use implementation details :-(
    t0StartTime = cache.data[0].expireTime - EXPIRE_TIME
    t1StartTime = cache.data[1].expireTime - EXPIRE_TIME
    self.assertTrue( t0StartTime < t1StartTime )
    self.assertTrue( t1StartTime < t0StartTime + CALC_TIME )
    self.assertAlmostEqual(t1StartTime-t0StartTime, T1_START_TIME, places=2)
    return


  def test_bg_cleanup(self):
    bgExecuter = BackgroundExecuter()
    
    EXPIRE_SECS = 0.1
    maxNumKeys = 2
    cache = cachelib.ExpiringMemCache(EXPIRE_SECS, maxNumKeys,
                                      selfCleanup=False,
                                      bgCleanup=bgExecuter)
    f = itertools.count().next

    self.assertEqual(cache.get_cached(0, f), 0)
    self.assertEqual(cache.get_cached(1, f), 1)
    self.assertEqual(cache.get_cached(2, f), 2)
    time.sleep(0.06)
    self.assertEqual(cache.get_cached(3, f), 3)

    self.assertEqual(len(cache), 4)
    bgExecuter.execute()
    self.assertEqual(len(cache), 2)
    assert 2 in cache
    assert 3 in cache
    
    time.sleep(0.06)
    self.assertEqual(len(cache), 2)
    assert 2 in cache
    assert 3 in cache
    bgExecuter.execute()
    
    self.assertEqual(len(cache), 1)
    assert 3 in cache

    time.sleep(0.06)
    self.assertEqual(len(cache), 1)
    assert 3 in cache
    self.assertEqual(cache.get_cached(0, f), 4)
    self.assertEqual(len(cache), 2)
    assert 3 in cache
    bgExecuter.execute()
    assert not 3 in cache

    del cache
    self.assertEqual(len(bgExecuter.tasks), 1)
    bgExecuter.execute()
    self.assertEqual(len(bgExecuter.tasks), 0)
    return
  
  #
  # Decorator tests
  #
  def test_dec_insert_expire(self):
    count = itertools.count()
    EXPIRE_SECS = 0.1

    @cachelib.ExpiringMemCache.cached(EXPIRE_SECS, None)
    def f(key, cnt, z):
      if not z=="z":
        raise KeyError(z)
      return cnt.next()

    kwargs = dict(z="z")
    
    self.assertRaises(KeyError, f, 0,count, z="a")

    self.assertEqual(f(0, count, **kwargs), 0)

    time.sleep(0.5*EXPIRE_SECS)
    self.assertEqual(f(0, count, **kwargs), 0)
    self.assertEqual(f(0, count, **kwargs), 0)
    self.assertEqual(f(1, count, **kwargs), 1)

    time.sleep(0.6*EXPIRE_SECS)
    self.assertEqual(f(0, count, **kwargs), 2)
    self.assertEqual(f(1, count, **kwargs), 1)

    time.sleep(0.6*EXPIRE_SECS)
    self.assertEqual(f(0, count, **kwargs), 2)
    self.assertEqual(f(1, count, **kwargs), 3)

    time.sleep(1.1*EXPIRE_SECS)
    self.assertEqual(f(0, count, **kwargs), 4)
    self.assertEqual(f(1, count, **kwargs), 5)

    return

  def test_dec_threaded(self):
    """Multiple threads using a cached function

    This test is all about timing. Note that there are 3 time parameters of
    interest: 1. start time of a query, 2. duration of calculation, 3. expire
    time.

    Consider the following diagram with 2 threads:

      T_0                   T_1
       |                     |
       | query key A         |
       |   calc result       |
       |         ^           |
       |         |           | query key B
       |      calc time      | return cached result
       |         |           |
       |         |           |
       |         v           |
       |   cache result      |
       |                     |
       | return result       |
       |                     |

    As shown above, a thread that queries first may gets its result later if
    another thread asks for a cached result. Note that results only expire
    relative to the *cache time* of the result, and *not* the start time of
    the associated query.
    """
    flock = threading.Lock()
    
    @cachelib.ExpiringMemCache.cached(expireSecs=1.0, maxNumKeys=None)
    def f(cnt, key):
      """Takes about 0.2s to calculate"""
      time.sleep(0.2)
      with flock:
        return count.next()

    def worker(slp, func, cnt, key, q):
      time.sleep(slp)
      res = func(cnt,key)
      q.put((key,res))
      return

    count = itertools.count()
    q = queue.Queue()

    # 0,0
    t0 = threading.Thread(target=worker, args=(0., f, count,0, q))  
    # 1,1 -- while t0 is busy calculating
    t1 = threading.Thread(target=worker, args=(0.1, f, count,1, q))
    # 0,0 -- wait for result,cached, returned before t1
    t2 = threading.Thread(target=worker, args=(0.15, f, count,0, q)) 
    # 0,0 -- no waiting for result, cached, returned before t1
    t3 = threading.Thread(target=worker, args=(0.22, f, count,0, q)) 
    # 0,2 -- expired
    t4 = threading.Thread(target=worker, args=(1.22, f, count,0, q)) 
    
    threads = [t0,t1,t2,t3,t4]
    for t in threads:
      t.daemon = True

    for t in threads:
      t.start()

    for t in threads:
      t.join()

    result = []
    sentinel = object()
    q.put(sentinel)    
    x = q.get()
    while not x is sentinel:
      result += [ x ]
      x = q.get()

    self.assertEqual(result, [(0,0), (0,0), (0,0), (1,1), (0,2)])
    return

  def test_shared_cached(self):
    flock = threading.Lock()
    glock = threading.Lock()

    EXPIRE_SECS = 1.0
    cache = cachelib.ExpiringMemCache(EXPIRE_SECS, maxNumKeys=4)

    
    @cachelib.cached(cache)
    def f(cnt, key):
      """Takes about 0.2s to calculate"""
      time.sleep(0.2)
      with flock:
        return count.next()

    @cachelib.cached(cache)
    def g(cnt, key):
      """Takes about 0.3s to calculate"""
      time.sleep(0.3)
      with glock:
        return count.next()

    count = itertools.count()

    self.assertEqual(f(count, 0), 0) # @ 0.0s
    self.assertEqual(g(count, 0), 1) # @ 0.2s
    self.assertEqual(f(count, 0), 0) # @ 0.5s
    self.assertEqual(f(count, 1), 2) # @ 0.5s
    self.assertEqual(g(count, 1), 3) # @ 0.7s

    self.assertEqual(len(cache), 4)

    time.sleep(0.25) # @ 1.0 s
    self.assertEqual(f(count, 0), 4) # @ 1.25s
    self.assertEqual(g(count, 0), 1) # @ 1.45s

    time.sleep(0.10) # @ 1.45 s
    self.assertEqual(g(count, 0), 5) # @ 1.55s
    self.assertEqual(len(cache), 4)

    return


#
# -----------------------------------------------------------------------------
# 
class TestParallelExpiringMemCache(unittest.TestCase):
  def test_multi(self):
    CALC_TIME = 1.0
    EXPIRE_SECS = 1.2

    flock = threading.Lock()
    def f(cnt, key):
      """Takes about 0.2s to calculate"""
      time.sleep(CALC_TIME)
      with flock:
        return count.next()

    count = itertools.count()
    cache = cachelib.ExpiringParallelMemCache(EXPIRE_SECS, maxNumKeys=4)

    t0 = time.time()
    key = range(5)
    pargs = [(count, i) for i in key]

    result = cache.get_cached_multi(key, f, pargs)
    t1 = time.time()

    assert set(result) == set(key)
    assert t1-t0 < 1.1*CALC_TIME # Serial would take 5 * CALC_TIME

    n = count.next()
    time.sleep(EXPIRE_SECS + 0.01)
    r1 = cache.get_cached(1, f, (count,1))
    r3 = cache.get_cached(3, f, (count,3))
    assert r1 == n+1
    assert r3 == n+2

    t2 = time.time()
    result = cache.get_cached_multi(key, f, pargs)
    t3 = time.time()
    assert result[1] == r1
    assert result[3] == r3

    for x in [0,2,4]:
      assert result[x] > r3
    assert t3-t2 < 1.1*CALC_TIME # Serial would take 3 * CALC_TIME
    return

    
#
# =============================================================================
# 
if __name__ == '__main__':
  unittest.main()
