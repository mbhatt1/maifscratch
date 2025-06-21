import DefaultTheme from 'vitepress/theme'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import './custom.css'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()
    
    const renderMermaidDiagrams = async () => {
      if (typeof window === 'undefined') return
      
      try {
        // Dynamic import of Mermaid
        const { default: mermaid } = await import('mermaid')
        
        // Initialize Mermaid
        mermaid.initialize({
          startOnLoad: false,
          theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default',
          themeVariables: {
            primaryColor: '#3c82f6',
            primaryTextColor: '#1f2937',
            primaryBorderColor: '#e5e7eb',
            lineColor: '#6b7280',
            secondaryColor: '#f3f4f6',
            tertiaryColor: '#ffffff'
          },
          flowchart: {
            useMaxWidth: true,
            htmlLabels: true
          },
          sequence: {
            useMaxWidth: true
          }
        })
        
        // Find all mermaid code blocks
        const mermaidBlocks = document.querySelectorAll('pre code.language-mermaid:not([data-processed])')
        
        for (let i = 0; i < mermaidBlocks.length; i++) {
          const block = mermaidBlocks[i]
          const content = block.textContent.trim()
          
          if (!content) continue
          
          try {
            // Create a unique ID
            const id = `mermaid-${Date.now()}-${i}`
            
            // Render the diagram
            const { svg } = await mermaid.render(id, content)
            
            // Create wrapper div
            const wrapper = document.createElement('div')
            wrapper.className = 'mermaid-diagram'
            wrapper.innerHTML = svg
            wrapper.style.textAlign = 'center'
            wrapper.style.margin = '2rem 0'
            
            // Replace the pre element
            const preElement = block.parentElement
            if (preElement && preElement.tagName === 'PRE') {
              preElement.replaceWith(wrapper)
            }
            
            block.setAttribute('data-processed', 'true')
          } catch (error) {
            console.error('Mermaid render error:', error)
            block.setAttribute('data-processed', 'true')
          }
        }
      } catch (error) {
        console.error('Failed to load Mermaid:', error)
      }
    }

    onMounted(() => {
      nextTick(() => {
        renderMermaidDiagrams()
      })
    })

    watch(
      () => route.path,
      () => {
        nextTick(() => {
          renderMermaidDiagrams()
        })
      }
    )
  }
} 