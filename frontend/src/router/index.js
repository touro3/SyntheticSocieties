import { createRouter, createWebHistory } from 'vue-router'
import HomeView        from '../views/HomeView.vue'
import RunView         from '../views/RunView.vue'
import MonitorView     from '../views/MonitorView.vue'
import ExperimentsView from '../views/ExperimentsView.vue'
import InteractView    from '../views/InteractView.vue'
import HumanEvalView   from '../views/HumanEvalView.vue'
import NotFoundView    from '../views/NotFoundView.vue'

// ResultsView imports Chart.js (~200 KB). Lazy-load it so the initial JS bundle
// stays small — the chart code is only downloaded when the user visits /results.
const ResultsView = () => import('../views/ResultsView.vue')

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/',                    component: HomeView },
    { path: '/run',                 component: RunView },
    { path: '/monitor/:expId',      component: MonitorView },
    { path: '/experiments',         component: ExperimentsView },
    { path: '/results/:expId',      component: ResultsView },
    { path: '/interact/:expId',     component: InteractView },
    { path: '/human-eval',          component: HumanEvalView },
    { path: '/:pathMatch(.*)*',     component: NotFoundView },
  ],
  scrollBehavior: () => ({ top: 0 }),
})
