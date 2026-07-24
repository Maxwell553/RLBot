const explicitOps = import.meta.env.VITE_ENABLE_OPS === 'true'
const explicitModeSwitch = import.meta.env.VITE_SHOW_MODE_SWITCH === 'true'

/**
 * The operations bundle is available automatically only in local development.
 * Production deployments must opt in explicitly and still rely on the
 * operator API's authentication; this flag is not an authorization boundary.
 */
export const OPS_UI_ENABLED = import.meta.env.DEV || explicitOps

/** Visual convenience for local/internal builds, never shown by default in production. */
export const MODE_SWITCH_VISIBLE = import.meta.env.DEV || explicitModeSwitch
