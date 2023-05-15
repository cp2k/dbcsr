/*

  Cloak
  https://github.com/pfultz2/Cloak
  A mini preprocessor library
  Copyright (c) 2012-2014 Paul Fultz II  
  
  Use, modification and distribution is subject to the Boost Software 
  License, Version 1.0. See the accompanying file LICENSE or

      http://www.booost.org/LICENSE_1_0.txt

  Permission granted by the author to include this file in vecLibFort
  under the terms of this license.

*/

#define CAT(a, ...) PRIMITIVE_CAT(a, __VA_ARGS__)
#define PRIMITIVE_CAT(a, ...) a ## __VA_ARGS__

#define INC(x) PRIMITIVE_CAT(INC_, x)
#define INC_0 1
#define INC_1 2
#define INC_2 3
#define INC_3 4
#define INC_4 5
#define INC_5 6
#define INC_6 7
#define INC_7 8
#define INC_8 9
#define INC_9 10
#define INC_10 11
#define INC_11 12

#define DEC(x) PRIMITIVE_CAT(DEC_, x)
#define DEC_0 0
#define DEC_1 0
#define DEC_2 1
#define DEC_3 2
#define DEC_4 3
#define DEC_5 4
#define DEC_6 5
#define DEC_7 6
#define DEC_8 7
#define DEC_9 8
#define DEC_10 9
#define DEC_11 10  

#define EXPR_S(s) PRIMITIVE_CAT(EXPR_, s)
#define EXPR_0(...) __VA_ARGS__
#define EXPR_1(...) __VA_ARGS__
#define EXPR_2(...) __VA_ARGS__
#define EXPR_3(...) __VA_ARGS__
#define EXPR_4(...) __VA_ARGS__
#define EXPR_5(...) __VA_ARGS__
#define EXPR_6(...) __VA_ARGS__
#define EXPR_7(...) __VA_ARGS__
#define EXPR_8(...) __VA_ARGS__
#define EXPR_9(...) __VA_ARGS__
#define EXPR_10(...) __VA_ARGS__
#define EXPR_11(...) __VA_ARGS__ 
#define EXPR_12(...) __VA_ARGS__
  
#define CHECK_N(x, n, ...) n
#define CHECK(...) CHECK_N(__VA_ARGS__, 0,)

#define NOT(x) CHECK(PRIMITIVE_CAT(NOT_, x))
#define NOT_0 ~, 1,

#define COMPL(b) PRIMITIVE_CAT(COMPL_, b)
#define COMPL_0 1
#define COMPL_1 0

#define BOOL(x) COMPL(NOT(x))

#define IIF(c) PRIMITIVE_CAT(IIF_, c)
#define IIF_0(t, ...) __VA_ARGS__
#define IIF_1(t, ...) t

#define IF(c) IIF(BOOL(c))

#define EAT(...)
#define EXPAND(...) __VA_ARGS__
#define WHEN(c) IF(c)(EXPAND, EAT)

#define EMPTY()
#define DEFER(id) id EMPTY()
#define OBSTRUCT(id) id DEFER(EMPTY)()
     
//#define REPEAT_S(s, n, m, ...) \
//        IF(n)(REPEAT_I, EAT)(OBSTRUCT(), INC(s), DEC(n), m, __VA_ARGS__)
//        
//#define REPEAT_INDIRECT() REPEAT_S
//#define REPEAT_I(_, s, n, m, ...) \
//        EXPR_S _(s)( \
//            REPEAT_INDIRECT _()(s, n, m, __VA_ARGS__) \
//        )\
//        m _(s, n, __VA_ARGS__)
        
#define REPEAT_S(s, n, m, ...) \
        REPEAT_I(OBSTRUCT(), INC(s), n, m, __VA_ARGS__)
        
#define REPEAT_INDIRECT() REPEAT_I
#define REPEAT_I(_, s, n, m, ...) \
        WHEN _(n)(EXPR_S _(s)( \
            REPEAT_INDIRECT _()(OBSTRUCT _(), INC _(s), DEC _(n), m, __VA_ARGS__) \
        ))\
        m _(s, n, __VA_ARGS__)

#define COMMA() ,

#define COMMA_IF(n) IF(n)(COMMA, EAT)()

