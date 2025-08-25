/*!
 * Copyright (c) 2026 Segno System.
 * @file    pd_role.h
 */

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace allspark::pd {

enum class PdRole {
  kColocated = 0,
  kPrefill = 1,
  kDecode = 2,
  kInvalid = 3,
};

inline PdRole ParsePdRole(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(),
                 [](unsigned char ch) { return std::tolower(ch); });
  if (value.empty() || value == "colocated" || value == "none") {
    return PdRole::kColocated;
  }
  if (value == "prefill") {
    return PdRole::kPrefill;
  }
  if (value == "decode") {
    return PdRole::kDecode;
  }
  return PdRole::kInvalid;
}

inline const char* PdRoleName(PdRole role) {
  switch (role) {
    case PdRole::kColocated:
      return "colocated";
    case PdRole::kPrefill:
      return "prefill";
    case PdRole::kDecode:
      return "decode";
    case PdRole::kInvalid:
      return "invalid";
  }
  return "invalid";
}

struct PdRuntimeConfig {
  PdRole role = PdRole::kColocated;
  std::string control_endpoint;

  bool IsSeparated() const {
    return role == PdRole::kPrefill || role == PdRole::kDecode;
  }

  bool IsValid() const {
    return role != PdRole::kInvalid &&
           (!IsSeparated() || !control_endpoint.empty());
  }
};

inline PdRuntimeConfig LoadPdRuntimeConfig() {
  const char* role = std::getenv("AS_PD_ROLE");
  const char* endpoint = std::getenv("AS_PD_CONTROL_ENDPOINT");
  return {ParsePdRole(role == nullptr ? "" : role),
          endpoint == nullptr ? "" : endpoint};
}

}  // namespace allspark::pd
