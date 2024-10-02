/**
 * Creating a sidebar enables you to:
 - create an ordered group of docs
 - render a sidebar for each doc of that group
 - provide next/previous navigation

 The sidebars can be generated from the filesystem, or explicitly defined here.

 Create as many sidebars as you want.
 */

// @ts-check

/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
    // By default, Docusaurus generates a sidebar from the docs folder structure
    // tutorialSidebar: [{ type: 'autogenerated', dirName: '.' }],

    // But you can create a sidebar manually

    // tutorialSidebar: [{
    //     type: 'category',
    //     label: 'React',
    //     items: ['react'],
    // }, ],

    build_dev_env: [
        {
            type: 'category',
            label: '介绍',
            items: ['index'],
        },
        {
            type: 'category',
            label: '预备知识',
            items: [
                {
                    type: 'autogenerated',
                    dirName: '00_prev_concept',
                },
            ],
        },
        {
            type: 'category',
            label: '新手村系列',
            items: [
                {
                    type: 'autogenerated',
                    dirName: '01_build_dev_env'
                },
                {
                    type: 'autogenerated',
                    dirName: '02_first_kernel'
                },
                {
                    type: 'autogenerated',
                    dirName: '03_nvprof_usage'
                },
                {
                    type: 'autogenerated',
                    dirName: '04_first_refine_kernel'
                },
                {
                    type: 'autogenerated',
                    dirName: '10_what_my_id'
                }
            ],
        },
        {
            type: 'category',
            label: '初级系列',
            items: [
                {
                    type: 'autogenerated',
                    dirName: '05_intro_parallel',
                },
                {
                    type: 'autogenerated',
                    dirName: '06_impl_matmul'
                },
                {
                    type: 'autogenerated',
                    dirName: '07_optimize_matmul'
                }
            ],
        },
        {
            type: 'category',
            label: '中级系列',
            items: [
                {
                    type: 'autogenerated',
                    dirName: '08_impl_reduce'
                },
                {
                    type: 'category',
                    label: 'Reduce 性能优化实践',
                    items: [
                        {
                            type: "autogenerated",
                            dirName: "09_optimize_reduce/01_interleaved_addressing"
                        },
                        {
                            type: "autogenerated",
                            dirName: "09_optimize_reduce/02_bank_conflict"
                        },
                        {
                            type: "autogenerated",
                            dirName: "09_optimize_reduce/03_idle_threads_free",
                        },
                        {
                            type: "autogenerated",
                            dirName: "09_optimize_reduce/04_unroll"
                        }
                    ]
                },
                {
                    type: 'category',
                    label: 'GEMM 优化专题',
                    items: [
                        {
                            type: "autogenerated",
                            dirName: "11_gemm_optimize/01_tiled2d"
                        },
                        {
                            type: "autogenerated",
                            dirName: "11_gemm_optimize/02_vectorize_smem_and_gmem_accesses"
                        },
                        {
                            type: "autogenerated",
                            dirName: "11_gemm_optimize/03_warptiling"
                        },
                        {
                            type: "autogenerated",
                            dirName: "11_gemm_optimize/04_double_buffer"
                        },
                        {
                            type: "autogenerated",
                            dirName: "11_gemm_optimize/05_bank_conflicts"
                        }
                    ] 
                },
                {
                    type: 'category',
                    label: '卷积优化专题',
                    items: [
                        {
                            type: "autogenerated",
                            dirName: "12_convolution/01_naive_conv"
                        },
                        {
                            type: "autogenerated",
                            dirName: "12_convolution/02_intro_conv_optimize"
                        },
                        {
                            type: "autogenerated",
                            dirName: "12_convolution/03_im2col_conv"
                        },
                        {
                            type: "autogenerated",
                            dirName: "12_convolution/04_implicit_gemm"
                        },
                        {
                            type: "autogenerated",
                            dirName: "12_convolution/05_cutlass_conv"
                        }
                    ]
                }
            ],
        },
        {
            type: 'category',
            label: 'LLM 推理技术',
            items: [
                {
                    type: 'autogenerated',
                    dirName: '13_continuous_batch'
                },
                {
                    type: 'autogenerated',
                    dirName: '14_page_attention'
                },
            ],
        }
    ]
};

module.exports = sidebars;
