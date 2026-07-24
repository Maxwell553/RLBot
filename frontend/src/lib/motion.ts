/** Product motion tokens — use these instead of ad-hoc durations/easings. */
export const MOTION = {
  fast: 0.15,
  base: 0.2,
  slow: 0.3,
  entrance: 0.65,
  ease: [0.16, 1, 0.3, 1] as const,
}

export const pageTransition = {
  initial: { opacity: 0, y: 8 },
  animate: { opacity: 1, y: 0 },
  exit: { opacity: 0, y: -5 },
  transition: { duration: MOTION.base },
}

export const reveal = {
  initial: { opacity: 0, y: 18 },
  whileInView: { opacity: 1, y: 0 },
  viewport: { once: true, margin: '-80px' },
  transition: { duration: MOTION.entrance, ease: MOTION.ease },
}
