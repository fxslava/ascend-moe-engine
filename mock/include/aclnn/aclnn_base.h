#pragma once
#include "acl_meta.h"
extern "C" {
aclnnStatus aclnnInit(const char*);
aclnnStatus aclnnFinalize();
}
