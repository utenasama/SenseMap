//Copyright (c) 2019, SenseTime Group.
//All rights reserved.

#ifndef SENSEMAP_UTIL_TESTING_H_
#define SENSEMAP_UTIL_TESTING_H_

#include <iostream>

#define BOOST_TEST_MAIN

#ifndef TEST_NAME
#error "TEST_NAME not defined"
#endif

#define BOOST_TEST_MODULE TEST_NAME

#include <boost/test/unit_test.hpp>

#endif