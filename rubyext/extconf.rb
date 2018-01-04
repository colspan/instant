require 'mkmf'
have_library("stdc++")
$INCFLAGS << " -I../"
$CPPFLAGS << "-std=c++14"
$DLDFLAGS << "-rdynamic ../build2/instant/libinstant.a -lmkldnn -lprotobuf"
create_makefile('instant')
