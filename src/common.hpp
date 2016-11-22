#ifndef _KAFFE_COMMON_H_
#define _KAFFE_COMMON_H_

#define INSTANTIATE_CLASS(classname) \
  template class classname<float>; \
  template class classname<double>

enum LOG_LEVEL {
  INFO,
  WARNING,
  ERROR,
  FATAL
};

class Logger {
};


static Logger logger;

template <typename T>
Logger& operator <<(Logger& l, T const& value) {
  return logger;
}

inline Logger& LOG(LOG_LEVEL level) {
  if ( level == ERROR || level == FATAL) {
    assert(false);
  }
  return logger;
}

inline Logger& CHECK(bool condition) {
  assert(condition);
  return logger;
}

template<typename T1, typename T2>
inline Logger& CHECK_EQ(T1 a, T2 b) {
  assert(a == b);
  return logger;
}

template<typename T1, typename T2>
inline Logger& CHECK_NE(T1 a, T2 b) {
  assert(a != b);
  return logger;
}

#endif
