import { createApp } from 'vue'
import App from './App.vue'
import router from './router/index.js'

const app = createApp(App)

/**
 * v-tilt — spring-physics 3D card tilt driven by mouse position.
 *
 * On mousemove: card tracks the cursor with a tiny CSS transition (fast follow).
 * On mouseleave: spring easing snaps the card back to flat (overshoot settle).
 * The spring feel comes from cubic-bezier(0.34, 1.56, 0.64, 1) on the return.
 */
app.directive('tilt', {
  mounted(el) {
    const PERSPECTIVE = 640
    const MAX_DEG = 9

    const onMove = (e) => {
      const r = el.getBoundingClientRect()
      const dx = (e.clientX - (r.left + r.width  / 2)) / (r.width  / 2)
      const dy = (e.clientY - (r.top  + r.height / 2)) / (r.height / 2)
      el.style.transition = 'transform 0.06s linear'
      el.style.transform  = `perspective(${PERSPECTIVE}px) rotateX(${-dy * MAX_DEG}deg) rotateY(${dx * MAX_DEG}deg) translateZ(8px) scale(1.015)`
    }

    const onLeave = () => {
      el.style.transition = 'transform 0.55s cubic-bezier(0.34, 1.56, 0.64, 1)'
      el.style.transform  = ''
    }

    el.addEventListener('mousemove', onMove, { passive: true })
    el.addEventListener('mouseleave', onLeave)
    el._tiltClean = () => {
      el.removeEventListener('mousemove', onMove)
      el.removeEventListener('mouseleave', onLeave)
    }
  },
  unmounted: el => el._tiltClean?.(),
})

app.use(router).mount('#app')
