/*!
 * Copyright (c) 2026 Segno System.
 * @file    pd_role_test.cpp
 */

#include <gtest/gtest.h>

#include <pd/pd_role.h>

namespace allspark::pd {
namespace {

TEST(PdRoleTest, ParsesSupportedRoles) {
  EXPECT_EQ(ParsePdRole(""), PdRole::kColocated);
  EXPECT_EQ(ParsePdRole("none"), PdRole::kColocated);
  EXPECT_EQ(ParsePdRole("COLOCATED"), PdRole::kColocated);
  EXPECT_EQ(ParsePdRole("Prefill"), PdRole::kPrefill);
  EXPECT_EQ(ParsePdRole("DECODE"), PdRole::kDecode);
}

TEST(PdRoleTest, RejectsUnknownRoles) {
  EXPECT_EQ(ParsePdRole("worker"), PdRole::kInvalid);
  EXPECT_STREQ(PdRoleName(PdRole::kInvalid), "invalid");
}

TEST(PdRoleTest, RequiresControlEndpointWhenSeparated) {
  PdRuntimeConfig colocated;
  EXPECT_TRUE(colocated.IsValid());
  EXPECT_FALSE(colocated.IsSeparated());

  PdRuntimeConfig missing_endpoint{PdRole::kPrefill, ""};
  EXPECT_TRUE(missing_endpoint.IsSeparated());
  EXPECT_FALSE(missing_endpoint.IsValid());

  PdRuntimeConfig decode{PdRole::kDecode, "unix:///tmp/dashinfer-pd.sock"};
  EXPECT_TRUE(decode.IsSeparated());
  EXPECT_TRUE(decode.IsValid());
}

}  // namespace
}  // namespace allspark::pd
