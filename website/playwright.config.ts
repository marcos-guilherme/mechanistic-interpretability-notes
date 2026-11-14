import {defineConfig, devices} from '@playwright/test';

const projectBaseUrl = 'http://127.0.0.1:4173/ioi-replication/';
const serverOrigin = 'http://127.0.0.1:4173';

export default defineConfig({
  testDir: './tests/e2e',
  fullyParallel: true,
  forbidOnly: Boolean(process.env.CI),
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 2 : undefined,
  reporter: process.env.CI ? 'github' : 'list',
  use: {
    baseURL: serverOrigin,
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run serve:project-page',
    url: projectBaseUrl,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [
    {
      name: 'desktop-chromium',
      use: {...devices['Desktop Chrome']},
    },
    {
      name: 'mobile-chromium',
      use: {
        ...devices['Desktop Chrome'],
        viewport: {width: 390, height: 844},
        isMobile: true,
      },
    },
  ],
});
