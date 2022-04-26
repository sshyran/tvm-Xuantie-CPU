/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#ifndef TVM_RELAY_ANALYSIS_IMGDNN_PROFILER_H_
#define TVM_RELAY_ANALYSIS_IMGDNN_PROFILER_H_
#ifdef BUILD_PNNA
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "CnnBasicTypes.hpp"
#include "CnnHwPassAdapter.hpp"
#include "CnnLog.h"
#include "CnnModelMapper.hpp"
#include "CnnSegmentStats.hpp"

class CSICnnProfiler : CSICnnSegmentStats {
 public:
  void init(const CSICnnModel* const model, const CSICnnHwOptimizerBase* const optimizer,
            int strategy_idx);
  void write_trace_data_json(const std::string& filename);

  /*! \brief Normal strategy to use ocm: if buffer size is less than available ocm size, then it
   * will be put into ocm. */
  void ocm_strategy1(CnnHwPassAdapter& pass, std::map<int, unsigned>& seg_used_ocm,       // NOLINT
                     int segment_idx,                                                     // NOLINT
                     const unsigned max_ocm_size, bool last_of_group, PassStats& stats);  // NOLINT

  /*! \brief Advanced strategy to use ocm: allocate memory in ocm according to
   * CnnPrioritizeMemPagesByHits.cpp pass. */
  void ocm_strategy2(const CSICnnModel* const model, CnnHwPassAdapter& pass,  // NOLINT
                     unsigned batch_idx,                                      // NOLINT
                     std::map<int, std::shared_ptr<std::vector<std::pair<unsigned, unsigned>>>>&
                         seg_pages_util,  // NOLINT
                     std::map<int, unsigned> seg_tmp_buffers, int segment_idx,
                     const unsigned page_size, const CSICnnHwOptimizerBase* const optimizer,
                     PassStats& stats);  // NOLINT

  unsigned get_io_ocm_saving(
      unsigned b_idx, unsigned base_addr, const CnnDims& shape, unsigned interleaving,
      unsigned bitdepth, unsigned bytes_per_burst, unsigned addr_align_bursts, unsigned page_size,
      std::shared_ptr<std::vector<std::pair<unsigned, unsigned>>>& seg_pages_util);  // NOLINT
};

#endif
#endif  // TVM_RELAY_ANALYSIS_IMGDNN_PROFILER_H_
