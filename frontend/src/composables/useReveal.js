import { onMounted, onBeforeUnmount } from 'vue'

/**
 * Staggered reveal driven by IntersectionObserver.
 * Adds `--i` CSS custom property (0-based index) and toggles `.revealed`
 * when the element enters the viewport.
 *
 * Usage:
 *   const container = ref(null)
 *   useReveal(container, '.reveal')
 *
 *   <div ref="container">
 *     <div class="reveal" v-for="...">...</div>
 *   </div>
 */
export function useReveal(containerRef, selector = '.reveal', opts = {}) {
  const { threshold = 0.08, rootMargin = '0px 0px -32px 0px' } = opts
  let observer = null

  onMounted(() => {
    const items = Array.from(containerRef.value?.querySelectorAll(selector) ?? [])
    items.forEach((el, i) => el.style.setProperty('--i', i))

    observer = new IntersectionObserver(
      entries => {
        entries.forEach(e => {
          if (e.isIntersecting) {
            e.target.classList.add('revealed')
            observer.unobserve(e.target)
          }
        })
      },
      { threshold, rootMargin }
    )
    items.forEach(el => observer.observe(el))
  })

  onBeforeUnmount(() => observer?.disconnect())
}
