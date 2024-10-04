// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require('prism-react-renderer/themes/github');
const darkCodeTheme = require('prism-react-renderer/themes/dracula');
const math = require('remark-math');
const katex = require('rehype-katex');

module.exports = {
    baseUrl: '/',
};

/** @type {import('@docusaurus/types').Config} */
const config = {
    title: 'Notebook',
    tagline: '',
    url: 'https://space.keter.top',
    baseUrl: '/',
    onBrokenLinks: 'throw',
    onBrokenMarkdownLinks: 'warn',
    favicon: 'img/favicon.ico',
    organizationName: 'facebook', // Usually your GitHub org/user name.
    projectName: 'CUDATutorial', // Usually your repo name.

    presets: [
        [
            'classic',
            /** @type {import('@docusaurus/preset-classic').Options} */
            ({
                docs: {
                    sidebarPath: require.resolve('./sidebars.js'),
                    // Please change this to your repo.
                    routeBasePath: '/',
                    editUrl: 'https://github.com/PaddleJitLab/CUDATutorial/tree/develop',
                    remarkPlugins: [math],
                    rehypePlugins: [katex],
                },
                blog: {
                    showReadingTime: true,
                    // Please change this to your repo.
                    routeBasePath: '/',
                    editUrl:
                        'https://github.com/PaddleJitLab/CUDATutorial/tree/develop',
                },
                theme: {
                    customCss: require.resolve('./src/css/custom.css'),
                },
            }),
        ],
    ],

    plugins: [
        [
            require.resolve("@cmfcmf/docusaurus-search-local"),
            {
                language: "zh",
            }
        ],
    ],

    themeConfig:
        /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
        ({
            metadata: [
                { name: 'keywords', content: 'CUDA' },
                { name: 'google-site-verification', content: 'rI6AJ6opyy43RCnFkrJtE5U-kufn37HuWdEVXmS43E' },
                { name: 'baidu-site-verification', content: 'code-BWReuturyA' }
            ],
            navbar: {
                title: 'CUDATutorial',
                logo: {
                    alt: 'My Site Logo',
                    src: 'img/logo.svg',
                },
                items: [
                    {
                        to: "/build_dev_env",
                        activeBasePath: '/build_dev_env',
                        label: "Docs",
                        position: "left",
                    },
                    {
                        href: 'https://github.com/PaddleJitLab/CUDATutorial',
                        label: 'GitHub',
                        position: 'right',
                    },
                ],
            },
            prism: {
                theme: lightCodeTheme,
                darkTheme: darkCodeTheme,
            },
        }),
    stylesheets: [
        {
            href: 'https://cdn.jsdelivr.net/npm/katex@0.13.24/dist/katex.min.css',
            type: 'text/css',
            integrity:
                'sha384-odtC+0UGzzFL/6PNoE8rX/SPcQDXBJ+uRepguP4QkPCm2LBxH3FA3y+fKSiJ+AmM',
            crossorigin: 'anonymous',
        },
    ],
};

module.exports = config;
