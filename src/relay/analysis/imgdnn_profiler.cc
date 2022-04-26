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
#ifdef BUILD_PNNA
#include "imgdnn_profiler.h"

#include <stdio.h>

#include <cstdlib>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "CnnHwPassAdapter.hpp"
#include "CnnLog.h"
#include "CnnModelMapper.hpp"
#include "CnnSegmentStats.hpp"

void CSICnnProfiler::init(const CSICnnModel* const model,
                          const CSICnnHwOptimizerBase* const optimizer, int strategy_idx) {
  // opts = optimizer->conf()->get_ints();
  auto& graph = model->get_graph();
  auto& blob_store = model->get_blob_store();
  std::list<unsigned> order = graph.linearize();

  /* Create a vector of layergroup parts.
   * Each part will be applied to each input in the batch to create a hardware pass
   */
  std::vector<std::tuple<CnnModelGraph::const_iterator, CnnHwPassAdapter, std::string>> lg_part;
  std::map<int /*segment idx*/,
           std::shared_ptr<std::vector<std::pair<unsigned, unsigned>>> /*mem page utilisation*/>
      seg_pages_util;
  std::map<int /*segment idx*/, unsigned /*page size*/> seg_page_size;
  std::map<int /*segment idx*/, unsigned /*page size*/> seg_page_max_count;
  std::map<int /*segment idx*/, unsigned /*tmp buffer id*/> seg_tmp_buffers;
  std::map<int /*segment idx*/, unsigned /*current used ocm*/> seg_used_ocm;
  // unsigned pre_group_id = 0;

  for (unsigned idx : order) {
    auto it_curr = graph.ct_at(idx);
    CnnConstMappingStepIterator it(it_curr);

    if (!it.have(kSegments)) continue;
    if (it.cdata<kSegments>().segment_idx < 0) continue;

    if (it.cdata<kSegments>().segment_begin && it.have(kMemPagePriority)) {
      if (seg_pages_util.find(it.cdata<kSegments>().segment_idx) == seg_pages_util.end()) {
        seg_page_size[it.cdata<kSegments>().segment_idx] = it.cdata<kMemPagePriority>().page_size;
        seg_page_max_count[it.cdata<kSegments>().segment_idx] =
            it.cdata<kMemPagePriority>().max_pages_ocm;
        seg_pages_util[it.cdata<kSegments>().segment_idx] = it.cdata<kMemPagePriority>().pages;
        seg_tmp_buffers[it.cdata<kSegments>().segment_idx] = it.cdata<kBuffOffs>().tmp_buff_id;

        seg_used_ocm[it.cdata<kSegments>().segment_idx] = 0;
      }
    }

    if (!CnnHwPassAdapter::first_of_pass(it.all_data())) continue;

    lg_part.emplace_back(it, CnnHwPassAdapter(graph, it), it_curr.id());
  }

  for (auto pass_info = lg_part.begin(); pass_info != lg_part.end(); ++pass_info) {
    CnnConstMappingStepIterator it(std::get<0>(*pass_info));
    try {
      CnnHwPassAdapter& pass = std::get<1>(*pass_info);

      CnnHwPassAdapter* prev_pass = nullptr;
      if (pass_info != lg_part.begin()) {
        prev_pass = &std::get<1>(*(pass_info - 1));
      }
      CnnHwPassAdapter* next_pass = nullptr;
      if ((pass_info + 1) != lg_part.end()) {
        next_pass = &std::get<1>(*(pass_info + 1));
      }

      const unsigned segment_idx = it.cdata<kSegments>().segment_idx;
      if (all_stats_.count(segment_idx) == 0) {
        all_stats_[segment_idx] = std::vector<PassStats>();
      }
      std::vector<PassStats>& passes_stats = all_stats_[segment_idx];

      /* This is assuming a fast moving B in all cases */
      const unsigned b_prime = 1;
      for (unsigned b = 0; b < pass.first().node()->input_shape.N(); b += b_prime) {
        PassStats stats;

        CnnConstMappingStepIterator first(pass.first());
        CnnConstMappingStepIterator last(pass.last());

        for (auto idx_in_pass : pass.ordered_nodes()) {
          auto idx_it = graph.ct_at(idx_in_pass);
          CnnConstMappingStepIterator tmp_it(idx_it);

          if (tmp_it.node()->type == UNKNOWN_NODE) {
            stats.ops.push_back(tmp_it.node()->custom_type);
          } else {
            stats.ops.push_back(CnnNode::type_to_string.at(tmp_it.type()));
          }
          stats.names.push_back(tmp_it.id());
          stats.orig_names.push_back(tmp_it.node()->original_id);
        }

        stats.seg_id = segment_idx;
        if (it.have(kDilationGroups))
          stats.group_id = it.cdata<kDilationGroups>().global_group_idx;
        else if (it.have(kConvSplits))
          stats.group_id = it.cdata<kConvSplits>().original_group_idx;
        else if (it.have(kEltwiseSplits))
          stats.group_id = it.cdata<kEltwiseSplits>().original_group_idx;
        else if (it.have(kPoolSplits))
          stats.group_id = it.cdata<kPoolSplits>().original_group_idx;
        else
          stats.group_id = it.cdata<kGroups>().group_idx;

        if (it.have(kConvSplits) && it.type() == CONVOLUTION_NODE) {
          stats.norm_overlap_begin = it.cdata<kConvSplits>().norm_overlap_begin;
          stats.norm_overlap_end = it.cdata<kConvSplits>().norm_overlap_end;
        } else if (it.have(kEltwiseSplits) && it.type() == ELTWISE_NODE) {
          stats.norm_overlap_begin = it.cdata<kEltwiseSplits>().norm_overlap_begin;
          stats.norm_overlap_end = it.cdata<kEltwiseSplits>().norm_overlap_end;
        }

        unsigned in_offset_x = 0;

        if (it.in_arcs().size() == 1) {
          std::string in_id = graph.ct_at(it.in_arcs().front()).id();
          auto in_it = graph.ct_at(in_id);
          if (in_it.type() == SPLIT_NODE &&
              as<CSICnnSplitNode>(in_it.node())->split_axis == WIDTH_AXIS) {
            auto branches = in_it.out_arcs();
            int idx = std::distance(branches.begin(),
                                    std::find(branches.begin(), branches.end(), it.idx()));
            in_offset_x = as<CSICnnSplitNode>(in_it.node())->split_indices[idx];
          } else if (in_it.type() == OVERLAP_NODE &&
                     as<CSICnnOverlapNode>(in_it.node())->split_axis == WIDTH_AXIS) {
            auto branches = in_it.out_arcs();
            int idx = std::distance(branches.begin(),
                                    std::find(branches.begin(), branches.end(), it.idx()));
            in_offset_x = as<CSICnnOverlapNode>(in_it.node())->parts[idx].first;
          }
        }

        CnnHwPassAdapter lg(graph, it.idx());
        auto last_d = lg.last().data();
        auto last_p = lg.last().node();

        stats.input_shape = first.node()->input_shape;
        stats.in_il = first.cdata<kInterleave>().in.interleaving;
        stats.in_bits = first.cdata<kFormats>().in_fmt.bits();
        stats.in_signed = first.cdata<kFormats>().in_fmt.is_signed();

/* @chenf: only used in csv writer, we can ignore them temporarily*/
#if 0
        if (first.have(kInBuffOffs) && first.have(kBuffOffs)) {
          if (first.cdata<kInBuffOffs>().load_input) {
            stats.read_offset = first.cdata<kBuffOffs>().in_base[0];
            stats.read_offset += first.cdata<kInBuffOffs>().in_offset;
          }
        } else {
          stats.read_offset = 0;
        }

        if (last.have(kInBuffOffs) && last.have(kBuffOffs)) {
          if (last.cdata<kInBuffOffs>().write_output)
            stats.write_offset = last.cdata<kBuffOffs>().out_base[0];
        } else {
          stats.write_offset = 0;
        }
#endif

        stats.out_il = data_of_<kInterleave>(last_d).out.interleaving;
        stats.output_shape = last_p->output_shape;
        stats.out_bits = data_of_<kFormats>(last_d).out_fmt.bits();
        stats.out_signed = data_of_<kFormats>(last_d).out_fmt.is_signed();
        stats.conv_output_shape = stats.input_shape;

        unsigned out_offset_x = 0;

        if (lg.last().out_arcs().size() == 1) {
          std::string merge_id = graph.ct_at(lg.last().out_arcs().front()).id();
          auto merge_it = graph.ct_at(merge_id);
          if (merge_it.type() == CONCAT_NODE &&
              as<CnnConcatNode>(merge_it.node())->concat_axis == WIDTH_AXIS) {
            for (unsigned i = 0; i < merge_it.in_arcs().size(); ++i) {
              if (merge_it.in_arcs_at(i) == lg.last().idx()) break;
              out_offset_x += graph.ct_at(merge_it.in_arcs_at(i)).node()->output_shape.W();
            }
          }
        }

        bool bias_term = false;
        bool load_input = true;
        bool load_coeff = (b == 0);
        bool load_accum = false, save_accum = false;
        unsigned weight_bits = 0, bias_bits = 0;

        if (pass.has_conv()) {
          auto conv_it = pass.conv();
          stats.conv = conv_it.id();
          stats.weights_shape = as<CSICnnConvNode>(conv_it.node())->blob_shapes[0];
          stats.bias_shape = as<CSICnnConvNode>(conv_it.node())->blob_shapes[1];
          bias_term = as<CSICnnConvNode>(conv_it.node())->bias_term;
          weight_bits = pass.weight_bits(blob_store);
          bias_bits = pass.bias_bits(blob_store);

          stats.conv_output_shape = pass.conv().node()->output_shape;
          stats.weight_bits = weight_bits;
          stats.bias_bits = bias_bits;
          stats.weights_signed = 1;

          if (as<CSICnnConvNode>(conv_it.node())->data.have(kAccum)) {
            load_accum = data_of_<kAccum>(as<CSICnnConvNode>(conv_it.node())->data).load_accum;
            save_accum = data_of_<kAccum>(as<CSICnnConvNode>(conv_it.node())->data).save_accum;
          }
          if (it.have(kBuffOffs)) {
            load_input = it.cdata<kBuffOffs>().load_input;
          }
          if (it.have(kConvSplits)) {
            load_coeff &= it.cdata<kConvSplits>().coeff_load;
          }

          layer_group_perf[stats.group_id].conv = (pass.conv().node()->original_id != "")
                                                      ? pass.conv().node()->original_id
                                                      : pass.conv().id();
        }

        if (pass.has_act()) {
          stats.act = pass.act().id();
          layer_group_perf[stats.group_id].act = (pass.act().node()->original_id != "")
                                                     ? pass.act().node()->original_id
                                                     : pass.act().id();
        }

        int tensorb_bitdepth = 0;
        int tensorb_il = 0;
        if (pass.has_eltwise()) {
          CnnConstMappingStepIterator eltwise_it(pass.eltwise());
          int b_idx = as<CnnEltwiseNode>(pass.eltwise().node())->tensorb_idx;
          if (b_idx >= 0) {
            unsigned b_graph_idx = eltwise_it.in_arcs_at(b_idx);
            while (graph.ct_at(b_graph_idx).in_arcs().size() > 0 &&
                   !model->isOp(graph.ct_at(b_graph_idx).type())) {
              b_graph_idx = graph.ct_at(b_graph_idx).in_arcs().front();
            }
            CnnConstMappingStepIterator b_it(graph.ct_at(b_graph_idx));
            CSIValueFormat b_fmt = b_it.cdata<kFormats>().have_lg_out_fmt
                                       ? b_it.cdata<kFormats>().lg_out_fmt
                                       : b_it.cdata<kFormats>().out_fmt;
            tensorb_bitdepth = b_fmt.bits();
            tensorb_il = b_it.cdata<kInterleave>().out.interleaving;
          } else if (!as<CnnEltwiseNode>(pass.eltwise().node())->blob_shape.empty()) {
            tensorb_bitdepth = data_of_<kCoefPacker>(pass.eltwise().data()).weights_fmt[0].bits();
            tensorb_il = data_of_<kCoefPacker>(pass.eltwise().data()).coef_il;
          }

          layer_group_perf[stats.group_id].ewo = (pass.eltwise().node()->original_id != "")
                                                     ? pass.eltwise().node()->original_id
                                                     : pass.eltwise().id();
        }

        if (pass.has_lrn()) {
          stats.lrn = pass.lrn().id();
          stats.has_acn = as<CnnLrnNode>(pass.lrn().node())->method == LrnMethod::ACROSS_CHANNELS;
          layer_group_perf[stats.group_id].lrn = (pass.lrn().node()->original_id != "")
                                                     ? pass.lrn().node()->original_id
                                                     : pass.lrn().id();
        }

        if (pass.has_pool()) {
          stats.pool = pass.pool().id();
          stats.pool_H = as<CnnPoolNode>(pass.pool().node())->kernel_size.dims[1];
          stats.pool_W = as<CnnPoolNode>(pass.pool().node())->kernel_size.dims[0];
          stats.pool_pad_t = as<CnnPoolNode>(pass.pool().node())->pad_begin.dims[0];
          stats.pool_pad_b = as<CnnPoolNode>(pass.pool().node())->pad_end.dims[0];
          stats.pool_stride_v = as<CnnPoolNode>(pass.pool().node())->stride.dims[0];
          stats.pool_stride_h = stats.pool_stride_v;

          layer_group_perf[stats.group_id].pool = (pass.pool().node()->original_id != "")
                                                      ? pass.pool().node()->original_id
                                                      : pass.pool().id();
        }

        // Making the assumption that B is always the fast moving split
        // this may not be teh case in future, but no way to assert this as yet
        auto usage = optimizer->pass_hw_usage(
            pass, prev_pass, next_pass, first.cdata<kInterleave>().in.interleaving,
            data_of_<kInterleave>(last_d).out.interleaving, tensorb_il,
            first.cdata<kFormats>().lg_in_fmt.bits(), data_of_<kFormats>(last_d).lg_out_fmt.bits(),
            weight_bits, bias_bits, tensorb_bitdepth, first.node()->input_shape, in_offset_x,
            stats.conv_output_shape, last_p->output_shape, out_offset_x, bias_term, load_accum,
            save_accum, load_input, load_coeff, const_cast<CSICnnBlobStore&>(blob_store));

        stats.coef_n = usage.coeff_r;         /* Coeffs needed - */
        stats.coef_r = usage.read_coef_bytes; /* Coeffs bytes read including prefetch - */
        stats.estimated_read_data_bytes = usage.read_data_bytes;
        stats.estimated_write_data_bytes = usage.write_data_bytes;
        stats.estimated_cycles = usage.cycles;

        stats.accum_w = usage.accum_w;
        stats.accum_r = usage.accum_r;
        stats.input_r = usage.input_r;
        stats.output_w = usage.output_w;

        bool out_fence = true;
        if (pass.has_conv()) {
          // CnnConstMappingStepIterator it(pass.conv());
          if (last.have(kOutputFence)) out_fence = last.cdata<kOutputFence>().fence_on;
        }

        bool last_of_group = false;
        if (next_pass == nullptr) {
          last_of_group = true;
        } else {
          CnnConstMappingStepIterator first_it_next(std::get<0>(*(pass_info + 1)));
          unsigned next_group_id;
          if (first_it_next.have(kConvSplits))
            next_group_id = first_it_next.cdata<kConvSplits>().original_group_idx;
          else if (first_it_next.have(kEltwiseSplits))
            next_group_id = first_it_next.cdata<kEltwiseSplits>().original_group_idx;
          else if (first_it_next.have(kPoolSplits))
            next_group_id = first_it_next.cdata<kPoolSplits>().original_group_idx;
          else
            next_group_id = first_it_next.cdata<kGroups>().group_idx;

          if (stats.group_id != next_group_id) last_of_group = true;
        }

        // // if output_fence is on, add 2*memory_latency, where memory_latency=256
        // if (out_fence && last_of_group) stats.estimated_cycles += 512;

        if (out_fence) stats.estimated_cycles += 512;

        // Calculate bandwidth savings due to OCM
        if (optimizer->conf()->ocm_enabled() && !pass.has_mmm()) {
          const int seg_idx = first.cdata<kSegments>().segment_idx;
          const unsigned page_size = seg_page_size[seg_idx];
          const unsigned max_page_count = seg_page_max_count[seg_idx];
          const unsigned max_ocm_size = page_size * max_page_count;

          switch (strategy_idx) {
            case 1:
              ocm_strategy1(pass, seg_used_ocm, seg_idx, max_ocm_size, last_of_group, stats);
              break;
            case 2:
              ocm_strategy2(model, pass, b, seg_pages_util, seg_tmp_buffers, segment_idx, page_size,
                            optimizer, stats);
              break;

            default:
              break;
          }
        }

        passes_stats.push_back(stats);

        layer_group_perf[stats.group_id].seg_id = segment_idx;

        layer_group_perf[stats.group_id].accum_w += stats.accum_w;
        layer_group_perf[stats.group_id].accum_r += stats.accum_r;
        layer_group_perf[stats.group_id].input_r += stats.input_r;
        layer_group_perf[stats.group_id].output_w += stats.output_w;
        layer_group_perf[stats.group_id].coef_r += stats.coef_r;
        layer_group_perf[stats.group_id].estimated_cycles += stats.estimated_cycles;

        for (auto item : stats.ops) {
          layer_group_perf[stats.group_id].ops.push_back(item);
        }
        for (auto item : stats.orig_names) {
          layer_group_perf[stats.group_id].orig_names.push_back(item);
        }
        for (auto item : stats.names) {
          layer_group_perf[stats.group_id].names.push_back(item);
        }

        totals_.estimated_cycles += stats.estimated_cycles;
        totals_.estimated_read_data_bytes += stats.estimated_read_data_bytes;
        totals_.estimated_write_data_bytes += stats.estimated_write_data_bytes;

        totals_.coef_r += stats.coef_r;

        // if (last_of_group) pre_group_id = stats.group_id;
      }
    } catch (CnnException& e) {
      log_err("error generating performance stats at %s", it.id().c_str());
      throw;
    }
  }

  for (auto& seg : all_stats_) {
    // startup overhead at beginning of every segment
    seg.second[0].estimated_cycles += 1500;
    layer_group_perf[seg.second[0].group_id].estimated_cycles += 1500;
    totals_.estimated_cycles += 1500;
  }

  log_debug(Stats, "-CnnProfiler::init");
}

void CSICnnProfiler::write_trace_data_json(const std::string& filename) {
  std::ofstream model_out(filename, ios::binary);
  CNN_CHECK(!model_out.fail(), "Failed to open layers file '%s'", filename.c_str());

  PnnaJson::Value header;
  PnnaJson::StyledStreamWriter writer;

  PnnaJson::Value buff_usage_json(PnnaJson::arrayValue);
  for (auto it = layer_group_perf.begin(); it != layer_group_perf.end(); ++it) {
    PnnaJson::Value layer_info;
    layer_info["cycles"] = it->second.estimated_cycles;
    layer_info["group_id"] = it->first;
    layer_info["segment_id"] = it->second.seg_id;
    layer_info["input_ddr"] = it->second.input_r;
    layer_info["output_ddr"] = it->second.output_w;
    layer_info["accum_ddr"] = it->second.accum_r + it->second.accum_w;
    layer_info["coeff_ddr"] = it->second.coef_r;

    PnnaJson::Value names_json(PnnaJson::arrayValue);
    for (auto item : it->second.names) {
      names_json.append(PnnaJson::Value(item));
    }
    layer_info["names"] = names_json;

    PnnaJson::Value ops_json(PnnaJson::arrayValue);
    for (auto item : it->second.ops) {
      ops_json.append(PnnaJson::Value(item));
    }
    layer_info["ops"] = ops_json;

    PnnaJson::Value orig_names_json(PnnaJson::arrayValue);
    for (auto item : it->second.orig_names) {
      orig_names_json.append(PnnaJson::Value(item));
    }
    layer_info["orig"] = orig_names_json;

    buff_usage_json.append(layer_info);
  }

  header["layers"] = buff_usage_json;

  std::ostringstream oss;
  writer.write(oss, header);
  std::string json_str = oss.str();

  model_out << json_str;

  model_out.close();
}

void CSICnnProfiler::ocm_strategy1(CnnHwPassAdapter& pass, std::map<int, unsigned>& seg_used_ocm,
                                   int segment_idx, const unsigned max_ocm_size, bool last_of_group,
                                   PassStats& stats) {
  CnnConstMappingStepIterator first(pass.first());
  CnnConstMappingStepIterator last(pass.last());
  unsigned release_in_pass = 0;

  if (first.cdata<kBuffOffs>().in_buffs.size() > 0 && first.cdata<kBuffOffs>().load_input) {
    if (stats.input_r <= (max_ocm_size - seg_used_ocm[segment_idx])) {
      seg_used_ocm[segment_idx] += stats.input_r;
      stats.estimated_read_data_bytes -= stats.input_r;
      release_in_pass += stats.input_r;
      stats.input_r = 0;
    }
  }

  if (pass.has_conv() && pass.conv().data().have(kAccum)) {
    CnnConstMappingStepIterator conv_it(pass.conv());
    if (conv_it.cdata<kAccum>().load_accum) {
      if (stats.accum_r <= (max_ocm_size - seg_used_ocm[segment_idx])) {
        seg_used_ocm[segment_idx] += stats.accum_r;
        stats.estimated_read_data_bytes -= stats.accum_r;
        release_in_pass += stats.accum_r;
        stats.accum_r = 0;
      }
    }

    if (conv_it.cdata<kAccum>().save_accum) {
      if (stats.accum_w <= (max_ocm_size - seg_used_ocm[segment_idx])) {
        seg_used_ocm[segment_idx] += stats.accum_w;
        stats.estimated_write_data_bytes -= stats.accum_w;
        release_in_pass += stats.accum_w;
        stats.accum_w = 0;
      }
    }
  }

  if (last.cdata<kBuffOffs>().out_buffs.size() > 0) {
    if (stats.output_w <= (max_ocm_size - seg_used_ocm[segment_idx])) {
      seg_used_ocm[segment_idx] += stats.output_w;
      stats.estimated_write_data_bytes -= stats.output_w;
      stats.output_w = 0;
    }
  }

  // if (pass.has_eltwise() && as<CnnEltwiseNode>(pass.eltwise().node())->tensorb_idx > -1) {
  //   CnnConstMappingStepIterator ewo_it(pass.eltwise());

  //   if (ewo_it.cdata<kBuffOffs>().tensorB_buffs[0].id ==
  //       seg_tmp_buffers[ewo_it.cdata<kSegments>().segment_idx]) {
  //     stats.input_r = stats.input_r > max_ocm_size ? stats.input_r : 0;
  //     stats.estimated_read_data_bytes = stats.estimated_read_data_bytes > max_ocm_size
  //                                           ? stats.estimated_read_data_bytes
  //                                           : 0;
  //   }
  // }

  if (!last_of_group) {
    seg_used_ocm[segment_idx] -= release_in_pass;
  } else {
    seg_used_ocm[segment_idx] = 0;
  }
}

void CSICnnProfiler::ocm_strategy2(
    const CSICnnModel* const model, CnnHwPassAdapter& pass, unsigned batch_idx,
    std::map<int, std::shared_ptr<std::vector<std::pair<unsigned, unsigned>>>>& seg_pages_util,
    std::map<int, unsigned> seg_tmp_buffers, int segment_idx, const unsigned page_size,
    const CSICnnHwOptimizerBase* const optimizer, PassStats& stats) {
  auto& graph = model->get_graph();
  CnnConstMappingStepIterator first(pass.first());
  CnnConstMappingStepIterator last(pass.last());

  auto& in_buffoffs_data = first.cdata<kBuffOffs>();
  if (first.have(kBuffOffs) && !in_buffoffs_data.in_buffs.empty() &&
      in_buffoffs_data.in_buffs[0].id == seg_tmp_buffers[first.cdata<kSegments>().segment_idx] &&
      in_buffoffs_data.load_input) {
    unsigned savings = 0;

    CnnDims& shape = first.node()->input_shape;
    if (pass.has_crop() && pass.has_conv()) {
      shape = as<CSICnnConvNode>(pass.conv().node())->input_shape;
    }
    const unsigned il = first.cdata<kInterleave>().in.interleaving;

    savings = get_io_ocm_saving(
        batch_idx, in_buffoffs_data.in_buffs[0].batch_base[batch_idx], shape, il,
        first.cdata<kFormats>().lg_in_fmt.bits(), optimizer->conf()->mem_burst_bytes(),
        optimizer->conf()->num_mem_burst_to_align(), page_size, seg_pages_util[segment_idx]);
    stats.input_r -= savings;
    stats.estimated_read_data_bytes -= savings;
  }

  auto& out_buffoffs_data = last.cdata<kBuffOffs>();
  if (last.have(kBuffOffs) && !out_buffoffs_data.out_buffs.empty() &&
      out_buffoffs_data.out_buffs[0].id == seg_tmp_buffers[last.cdata<kSegments>().segment_idx]) {
    unsigned savings = 0;

    CnnDims& shape = last.node()->output_shape;
    if (pass.has_pool() && last.have(kPoolSplits)) {
      shape.num_channels() = last.cdata<kPoolSplits>().outpack_planes;
    }
    if (pass.has_crop() && pass.has_conv()) {
      shape = as<CSICnnConvNode>(pass.conv().node())->output_shape;
    }
    const unsigned il = last.cdata<kInterleave>().out.interleaving;

    savings = get_io_ocm_saving(
        batch_idx, out_buffoffs_data.out_buffs[0].batch_base[batch_idx], shape, il,
        last.cdata<kFormats>().lg_out_fmt.bits(), optimizer->conf()->mem_burst_bytes(),
        optimizer->conf()->num_mem_burst_to_align(), page_size, seg_pages_util[segment_idx]);
    stats.output_w -= savings;
    stats.estimated_write_data_bytes -= savings;
  }

  if (pass.has_conv() && pass.conv().data().have(kAccum)) {
    CnnConstMappingStepIterator conv_it(pass.conv());
    const CnnDims& out_shape = conv_it.node()->output_shape;

    unsigned accum_line_stride = RoundUp(out_shape.P() * out_shape.W() * 4,
                                         optimizer->conf()->addr_alignment_byte());  // float32
    unsigned bursts_per_line = DivideUp(out_shape.W() * out_shape.P() * 4, 32) >> 2;

    if (conv_it.cdata<kAccum>().load_accum) {
      unsigned savings = 0;

      for (unsigned y = 0; y < out_shape.H(); ++y) {
        for (unsigned burst = 0; burst < bursts_per_line; ++burst) {
          unsigned addr = conv_it.cdata<kBuffOffs>().load_accum_buff.batch_base[batch_idx] +
                          y * accum_line_stride + burst * optimizer->conf()->mem_burst_bytes();

          for (auto page : *(seg_pages_util[segment_idx])) {
            unsigned page_addr = page.first * page_size;

            if (addr >= page_addr && addr < page_addr + page_size && page.second > 0) {
              savings += optimizer->conf()->mem_burst_bytes();
              page.second -= 1;
            }
          }
        }
      }
      stats.accum_r -= savings;
      stats.estimated_read_data_bytes -= savings;
    }

    if (conv_it.cdata<kAccum>().save_accum) {
      unsigned savings = 0;

      for (unsigned y = 0; y < out_shape.H(); ++y) {
        for (unsigned burst = 0; burst < bursts_per_line; ++burst) {
          unsigned addr = conv_it.cdata<kBuffOffs>().store_accum_buff.batch_base[batch_idx] +
                          y * accum_line_stride + burst * optimizer->conf()->mem_burst_bytes();

          for (auto page : *(seg_pages_util[segment_idx])) {
            unsigned page_addr = page.first * page_size;

            if (addr >= page_addr && addr < page_addr + page_size && page.second > 0) {
              savings += optimizer->conf()->mem_burst_bytes();
              page.second -= 1;
            }
          }
        }
      }
      stats.accum_w -= savings;
      stats.estimated_write_data_bytes -= savings;
    }
  }

  if (pass.has_eltwise() && as<CnnEltwiseNode>(pass.eltwise().node())->tensorb_idx > -1) {
    CnnConstMappingStepIterator ewo_it(pass.eltwise());

    if (!ewo_it.cdata<kBuffOffs>().tensorB_buffs.empty() &&
        ewo_it.cdata<kBuffOffs>().tensorB_buffs[0].id ==
            seg_tmp_buffers[ewo_it.cdata<kSegments>().segment_idx]) {
      unsigned savings = 0;

      int b_idx = as<CnnEltwiseNode>(pass.eltwise().node())->tensorb_idx;
      CnnDims& shape = ewo_it.node()->input_shape;
      unsigned il;
      CSIValueFormat fmt;

      // Find spatial overlap node to get shape right
      std::string spatial_overlap_id;
      unsigned previous_idx = pass.eltwise().in_arcs_at(b_idx);
      bool backtrack = true;

      while (backtrack) {
        CnnConstMappingStepIterator previous_it(graph.ct_at(previous_idx));
        if (previous_it.type() == OVERLAP_NODE &&
            as<CSICnnOverlapNode>(previous_it.node())->split_axis == WIDTH_AXIS) {
          spatial_overlap_id = previous_it.id();
          backtrack = false;
        } else if ((previous_it.category() == TypeCategory::NOP ||
                    previous_it.type() == CROP_NODE) &&
                   previous_it.in_arcs().size() == 1) {
          previous_idx = previous_it.in_arcs_at(0);
        } else {
          backtrack = false;
        }

        if (!backtrack) {
          il = previous_it.cdata<kInterleave>().out.interleaving;
          fmt = previous_it.cdata<kFormats>().have_lg_out_fmt
                    ? previous_it.cdata<kFormats>().lg_out_fmt
                    : previous_it.cdata<kFormats>().out_fmt;
        }
      }
      if (spatial_overlap_id != "") {
        CnnConstMappingStepIterator previous_it(graph.ct_at(previous_idx));
        shape = graph.ct_at(spatial_overlap_id).node()->input_shape;
      }

      savings = get_io_ocm_saving(
          batch_idx, ewo_it.cdata<kBuffOffs>().tensorB_buffs[0].batch_base[batch_idx], shape, il,
          fmt.bits(), optimizer->conf()->mem_burst_bytes(),
          optimizer->conf()->num_mem_burst_to_align(), page_size, seg_pages_util[segment_idx]);
      stats.input_r -= savings;
      stats.estimated_read_data_bytes -= savings;
    }
  }
}

unsigned CSICnnProfiler::get_io_ocm_saving(
    unsigned b_idx, unsigned base_addr, const CnnDims& shape, unsigned interleaving,
    unsigned bitdepth, unsigned bytes_per_burst, unsigned addr_align_bursts, unsigned page_size,
    std::shared_ptr<std::vector<std::pair<unsigned, unsigned>>>& seg_pages_util) {
  unsigned savings = 0;

  const unsigned il = interleaving;

  const unsigned data_per_burst = RoundDown(8 * bytes_per_burst / bitdepth, 4);
  const unsigned bursts_per_line = DivideUp(shape.W() * il, data_per_burst);

  const unsigned line_stride = RoundUp(bursts_per_line, addr_align_bursts) * bytes_per_burst;
  const unsigned layer_stride = line_stride * shape.H();

  for (unsigned p = 0; p < DivideUp(shape.P(), il); ++p) {
    for (unsigned h = 0; h < shape.H(); ++h) {
      for (unsigned burst = 0; burst < bursts_per_line; ++burst) {
        unsigned addr = base_addr + b_idx * layer_stride * DivideUp(shape.P(), il) +
                        p * layer_stride + h * line_stride + burst * bytes_per_burst;

        for (auto& page : *(seg_pages_util)) {
          unsigned page_addr = page.first * page_size;

          if (addr >= page_addr && addr < page_addr + page_size && page.second > 0) {
            savings += bytes_per_burst;
            page.second -= 1;
          }
        }
      }
    }
  }

  return savings;
}
#endif
