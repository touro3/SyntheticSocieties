import { createRouter, createWebHashHistory } from 'vue-router'
import HomeView        from '../views/HomeView.vue'
import RunView         from '../views/RunView.vue'
import MonitorView     from '../views/MonitorView.vue'
import ExperimentsView from '../views/ExperimentsView.vue'
import ResultsView     from '../views/ResultsView.vue'
import InteractView    from '../views/InteractView.vue'
import HumanEvalView   from '../views/HumanEvalView.vue'

export default createRouter({
  history: createWebHashHistory(),
  routes: [
    { path: '/',                    component: HomeView },
    { path: '/run',                 component: RunView },
    { path: '/monitor/:expId',      component: MonitorView },
    { path: '/experiments',         component: ExperimentsView },
    { path: '/results/:expId',      component: ResultsView },
    { path: '/interact/:expId',     component: InteractView },
    { path: '/human-eval',          component: HumanEvalView },
  ],
  scrollBehavior: () => ({ top: 0 }),
})
