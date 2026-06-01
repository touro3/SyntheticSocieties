<template>
  <div class="not-found">
    <span class="code">404</span>
    <p>Page not found.</p>
    <router-link to="/" class="home-link">← Back to home</router-link>
  </div>
</template>

<style scoped>
.not-found {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  height: 60vh;
  gap: 12px;
  color: var(--text2, #aaa);
}
.code {
  font-size: 4rem;
  font-weight: 700;
  color: var(--text1, #eee);
  letter-spacing: -2px;
}
.home-link {
  color: var(--blue2, #5b9cf6);
  text-decoration: none;
  font-size: .9rem;
}
.home-link:hover { text-decoration: underline; }
</style>
