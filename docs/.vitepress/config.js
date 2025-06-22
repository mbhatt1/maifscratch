import { defineConfig } from 'vitepress'
import { withMermaid } from 'vitepress-plugin-mermaid'

export default withMermaid(defineConfig({
  title: 'MAIF Framework',
  description: 'Multi-Agent Intelligence Framework - Cutting-edge memory framework for AI agent systems with advanced privacy, semantic understanding, and high-performance capabilities',
  
  head: [
    ['meta', { name: 'theme-color', content: '#3c82f6' }],
    ['meta', { name: 'og:type', content: 'website' }],
    ['meta', { name: 'og:locale', content: 'en' }],
    ['meta', { name: 'og:site_name', content: 'MAIF Framework' }],
    ['meta', { name: 'og:image', content: '/maif-og-image.png' }],
    ['link', { rel: 'icon', href: '/favicon.ico' }],
    ['link', { rel: 'mask-icon', href: '/safari-pinned-tab.svg', color: '#3c82f6' }],
    ['meta', { name: 'msapplication-TileColor', content: '#3c82f6' }],
    // Custom CSS for code overflow and styling
    ['style', {}, `
      .vp-code-group .tabs { overflow-x: auto; }
      .vp-code-group .tabs::-webkit-scrollbar { height: 4px; }
      .vp-code-group .tabs::-webkit-scrollbar-thumb { background: var(--vp-c-divider); border-radius: 2px; }
      .language-python, .language-javascript, .language-typescript, .language-bash { 
        overflow-x: auto; 
        word-wrap: break-word; 
      }
      pre code { 
        white-space: pre; 
        overflow-x: auto; 
        display: block; 
        padding: 1rem; 
      }
      .mermaid-rendered { 
        text-align: center; 
        margin: 2rem 0; 
        overflow-x: auto; 
      }
      .mermaid-error {
        color: #ef4444;
        background: #fef2f2;
        border: 1px solid #fecaca;
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
      }
      @media (max-width: 768px) {
        pre code { font-size: 12px; }
        .mermaid-rendered { font-size: 12px; }
      }
    `]
  ],

  cleanUrls: true,
  lastUpdated: true,
  ignoreDeadLinks: true,
  
  // Markdown configuration with Mermaid support
  markdown: {
    theme: {
      light: 'github-light',
      dark: 'github-dark'
    },
    lineNumbers: true
  },

  // Mermaid configuration
  mermaid: {
    theme: 'default',
    flowchart: {
      useMaxWidth: false,
      htmlLabels: true,
      nodeSpacing: 50,
      rankSpacing: 50
    },
    sequence: {
      useMaxWidth: false,
      diagramMarginX: 50,
      diagramMarginY: 50,
      actorMargin: 50,
      width: 150,
      height: 65,
      boxMargin: 10,
      boxTextMargin: 5,
      noteMargin: 10,
      messageMargin: 35
    },
    gantt: {
      useMaxWidth: false
    },
    journey: {
      useMaxWidth: false
    },
    class: {
      useMaxWidth: false
    },
    state: {
      useMaxWidth: false
    },
    er: {
      useMaxWidth: false
    }
  },
  
  themeConfig: {
    logo: '/maif-logo.svg',
    siteTitle: 'MAIF Framework',
    
    nav: [
      { text: 'Guide', link: '/guide/getting-started' },
      { text: 'API Reference', link: '/api/' },
      { text: 'Examples', link: '/examples/' },
      {
        text: 'v1.0.0',
        items: [
          { text: 'Changelog', link: '/changelog' },
          { text: 'Contributing', link: '/contributing' }
        ]
      }
    ],

    sidebar: {
      '/guide/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/guide/getting-started' },
            { text: 'Installation', link: '/guide/installation' },
            { text: 'Quick Start', link: '/guide/quick-start' },
            { text: 'Core Concepts', link: '/guide/concepts' }
          ]
        },
        {
          text: 'Architecture',
          items: [
            { text: 'System Overview', link: '/guide/architecture' },
            { text: 'Block Structure', link: '/guide/blocks' },
            { text: 'Security Model', link: '/guide/security-model' },
            { text: 'Privacy Framework', link: '/guide/privacy' }
          ]
        },
        {
          text: 'Agent Development',
          items: [
            { text: 'Agent Lifecycle', link: '/guide/agent-lifecycle' },
            { text: 'Multi-modal Data', link: '/guide/multimodal' },
            { text: 'Semantic Understanding', link: '/guide/semantic' },
            { text: 'Real-time Processing', link: '/guide/streaming' }
          ]
        },
        {
          text: 'Advanced Topics',
          items: [
            { text: 'ACID Transactions', link: '/guide/acid' },
            { text: 'Performance Optimization', link: '/guide/performance' },
            { text: 'Distributed Deployment', link: '/guide/distributed' },
            { text: 'Monitoring & Observability', link: '/guide/monitoring' }
          ]
        }
      ],
      '/api/': [
        {
          text: 'Core API',
          items: [
            { text: 'Overview', link: '/api/' },
            { text: 'MAIFClient', link: '/api/core/client' },
            { text: 'Artifact', link: '/api/core/artifact' },
            { text: 'Encoder/Decoder', link: '/api/core/encoder-decoder' }
          ]
        },
        {
          text: 'Privacy & Security',
          items: [
            { text: 'Privacy Engine', link: '/api/privacy/engine' },
            { text: 'Security', link: '/api/security/index' },
            { text: 'Access Control', link: '/api/security/access-control' },
            { text: 'Cryptography', link: '/api/security/crypto' }
          ]
        }
      ],
      '/examples/': [
        {
          text: 'Basic Examples',
          items: [
            { text: 'Overview', link: '/examples/' },
            { text: 'Hello World Agent', link: '/examples/hello-world' },
            { text: 'Privacy-Enabled Agent', link: '/examples/privacy-agent' },
            { text: 'Multi-modal Processing', link: '/examples/multimodal' }
          ]
        },
        {
          text: 'Real-world Use Cases',
          items: [
            { text: 'Financial AI Agent', link: '/examples/financial-agent' },
            { text: 'Healthcare AI Agent', link: '/examples/healthcare-agent' },
            { text: 'Content Moderation', link: '/examples/content-moderation' },
            { text: 'Research Assistant', link: '/examples/research-assistant' },
            { text: 'Security Monitor', link: '/examples/security-monitor' }
          ]
        },
        {
          text: 'Integration Examples',
          items: [
            { text: 'LangChain Integration', link: '/examples/langchain' },
            { text: 'Hugging Face Models', link: '/examples/huggingface' },
            { text: 'Ray/Dask Distributed', link: '/examples/distributed' },
            { text: 'Kafka Streaming', link: '/examples/kafka' }
          ]
        }
      ]
    },

    editLink: {
      pattern: 'https://github.com/maif-ai/maif/edit/main/docs/:path'
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/maif-ai/maif' },
      { icon: 'discord', link: 'https://discord.gg/maif' },
      { icon: 'twitter', link: 'https://twitter.com/maif_ai' }
    ],

    footer: {
      message: 'Released under the MIT License.',
      copyright: 'Copyright © 2024 MAIF Contributors'
    },

    search: {
      provider: 'local',
      options: {
        translations: {
          button: {
            buttonText: 'Search MAIF docs'
          }
        }
      }
    }
  },

  vite: {
    define: {
      __VUE_OPTIONS_API__: false
    },
    optimizeDeps: {
      include: [
        '@vue/repl/codemirror-editor',
        '@vue/repl/monaco-editor',
        'mermaid'
      ]
    },
    // Performance optimizations
    build: {
      chunkSizeWarningLimit: 1000
    }
  },

  vue: {
    reactivityTransform: true
  },

  // Add client-side Mermaid initialization
  buildEnd() {
    // This will be handled by the theme
  },

  // PWA configuration
  pwa: {
    outDir: '.vitepress/dist',
    registerType: 'autoUpdate',
    includeAssets: ['favicon.ico', 'apple-touch-icon.png'],
    manifest: {
      name: 'MAIF Framework Documentation',
      short_name: 'MAIF Docs',
      description: 'Multi-Agent Intelligence Framework Documentation',
      theme_color: '#3c82f6',
      icons: [
        {
          src: 'pwa-192x192.png',
          sizes: '192x192',
          type: 'image/png'
        },
        {
          src: 'pwa-512x512.png',
          sizes: '512x512',
          type: 'image/png'
        }
      ]
    }
  }
})) 