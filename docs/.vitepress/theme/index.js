import DefaultTheme from 'vitepress/theme'
import { onMounted, watch, nextTick } from 'vue'
import { useRoute } from 'vitepress'
import './custom.css'

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()
    
    const initMermaid = async () => {
      if (typeof window === 'undefined') {
        console.log('[Mermaid] Skipping - running on server')
        return null
      }
      
      console.log('[Mermaid] Initializing...')
      
      try {
        const { default: mermaid } = await import('mermaid')
        console.log('[Mermaid] Module loaded successfully')
        
        // Initialize mermaid with proper configuration
        mermaid.initialize({
          startOnLoad: false,
          theme: document.documentElement.classList.contains('dark') ? 'dark' : 'default',
          securityLevel: 'loose',
          flowchart: {
            useMaxWidth: true,
            htmlLabels: true,
            curve: 'basis'
          },
          sequence: {
            useMaxWidth: true,
            wrap: true
          },
          gantt: {
            useMaxWidth: true
          },
          journey: {
            useMaxWidth: true
          }
        })
        
        console.log('[Mermaid] Initialized successfully')
        return mermaid
      } catch (error) {
        console.error('[Mermaid] Failed to load:', error)
        return null
      }
    }
    
    const renderMermaidDiagrams = async () => {
      if (typeof window === 'undefined') {
        console.log('[Mermaid] Skipping render - running on server')
        return
      }
      
      console.log('[Mermaid] Starting diagram rendering...')
      
      // Debug: Let's see what code blocks are actually in the DOM
      const allCodeBlocks = document.querySelectorAll('pre code')
      console.log(`[Mermaid] Found ${allCodeBlocks.length} total code blocks`)
      
      allCodeBlocks.forEach((block, index) => {
        console.log(`[Mermaid] Code block ${index + 1}:`, {
          className: block.className,
          classList: Array.from(block.classList),
          textContent: block.textContent?.substring(0, 50) + '...'
        })
      })
      
      const mermaid = await initMermaid()
      if (!mermaid) {
        console.error('[Mermaid] Cannot render - mermaid not initialized')
        return
      }
      
      // Try multiple selectors to find mermaid code blocks
      const selectors = [
        'pre code.language-mermaid',
        'pre code[class*="language-mermaid"]',
        'pre code[class*="mermaid"]',
        'code.language-mermaid',
        'code[class*="language-mermaid"]',
        'code[class*="mermaid"]'
      ]
      
      let mermaidCodeBlocks = []
      
      for (const selector of selectors) {
        const blocks = document.querySelectorAll(selector)
        if (blocks.length > 0) {
          console.log(`[Mermaid] Selector "${selector}" found ${blocks.length} blocks`)
          mermaidCodeBlocks = Array.from(blocks)
          break
        } else {
          console.log(`[Mermaid] Selector "${selector}" found 0 blocks`)
        }
      }
      
      console.log(`[Mermaid] Total mermaid code blocks found: ${mermaidCodeBlocks.length}`)
      
      if (mermaidCodeBlocks.length === 0) {
        console.log('[Mermaid] No mermaid diagrams found - trying fallback approach')
        
        // Fallback: look for any code block containing mermaid-like content
        const allBlocks = Array.from(document.querySelectorAll('pre code'))
        mermaidCodeBlocks = allBlocks.filter(block => {
          const content = block.textContent || ''
          const isMermaid = content.trim().match(/^(graph|flowchart|sequenceDiagram|classDiagram|stateDiagram|erDiagram|journey|gantt|pie|gitGraph)/i)
          if (isMermaid) {
            console.log('[Mermaid] Found mermaid content by content analysis:', content.substring(0, 50))
          }
          return isMermaid
        })
        
        console.log(`[Mermaid] Fallback found ${mermaidCodeBlocks.length} potential mermaid blocks`)
      }
      
      if (mermaidCodeBlocks.length === 0) {
        console.log('[Mermaid] Still no mermaid diagrams found after fallback')
        return
      }
      
      for (let i = 0; i < mermaidCodeBlocks.length; i++) {
        const codeBlock = mermaidCodeBlocks[i]
        console.log(`[Mermaid] Processing block ${i + 1}/${mermaidCodeBlocks.length}`)
        
        // Skip if already processed
        if (codeBlock.hasAttribute('data-mermaid-processed')) {
          console.log(`[Mermaid] Block ${i + 1} already processed, skipping`)
          continue
        }
        
        const content = codeBlock.textContent || codeBlock.innerText
        if (!content.trim()) {
          console.log(`[Mermaid] Block ${i + 1} is empty, skipping`)
          continue
        }
        
        console.log(`[Mermaid] Block ${i + 1} content:`, content.trim())
        
        try {
          // Create unique ID
          const id = `mermaid-diagram-${Date.now()}-${i}`
          console.log(`[Mermaid] Rendering diagram with ID: ${id}`)
          
          // Create container div
          const container = document.createElement('div')
          container.className = 'mermaid-container'
          container.style.cssText = `
            text-align: center;
            margin: 2rem 0;
            padding: 1rem;
            border: 1px solid var(--vp-c-divider);
            border-radius: 8px;
            background: var(--vp-c-bg);
            overflow-x: auto;
          `
          
          // Render the diagram
          const { svg } = await mermaid.render(id, content.trim())
          console.log(`[Mermaid] Successfully rendered diagram ${i + 1}`)
          container.innerHTML = svg
          
          // Replace the pre element with the rendered diagram
          const preElement = codeBlock.closest('pre')
          if (preElement && preElement.parentNode) {
            preElement.parentNode.replaceChild(container, preElement)
            console.log(`[Mermaid] Replaced pre element for diagram ${i + 1}`)
          } else {
            console.error(`[Mermaid] Could not find parent pre element for diagram ${i + 1}`)
          }
          
          // Mark as processed
          codeBlock.setAttribute('data-mermaid-processed', 'true')
          
        } catch (error) {
          console.error(`[Mermaid] Rendering error for diagram ${i + 1}:`, error)
          
          // Create error display
          const errorDiv = document.createElement('div')
          errorDiv.className = 'mermaid-error'
          errorDiv.style.cssText = `
            color: #ef4444;
            background: #fef2f2;
            border: 1px solid #fecaca;
            border-radius: 8px;
            padding: 1rem;
            margin: 1rem 0;
            font-family: monospace;
          `
          errorDiv.textContent = `Mermaid Error: ${error.message}`
          
          const preElement = codeBlock.closest('pre')
          if (preElement && preElement.parentNode) {
            preElement.parentNode.replaceChild(errorDiv, preElement)
            console.log(`[Mermaid] Replaced pre element with error for diagram ${i + 1}`)
          }
          
          codeBlock.setAttribute('data-mermaid-processed', 'true')
        }
      }
      
      console.log('[Mermaid] Finished processing all diagrams')
    }
    
    // Render on mount
    onMounted(() => {
      console.log('[Mermaid] Component mounted, scheduling diagram rendering')
      nextTick(() => {
        setTimeout(() => {
          console.log('[Mermaid] Executing scheduled rendering')
          renderMermaidDiagrams()
        }, 500) // Increased timeout to ensure DOM is fully ready
      })
    })
    
    // Re-render on route change
    watch(
      () => route.path,
      (newPath) => {
        console.log(`[Mermaid] Route changed to: ${newPath}, scheduling re-render`)
        nextTick(() => {
          setTimeout(() => {
            console.log('[Mermaid] Executing scheduled re-render')
            renderMermaidDiagrams()
          }, 500) // Increased timeout
        })
      }
    )
  }
} 