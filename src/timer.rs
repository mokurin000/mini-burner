//! Utility to measure and log elapsed time.

use std::time::Instant;

use spdlog::info;

/// Logs elapsed time when dropped.
///
/// Records the time at creation, then logs how long it lived.
pub struct Timer(Instant, &'static str);

impl Timer {
    /// Create a timer with a custom label.
    pub fn with_tip(tip: &'static str) -> Self {
        Self(Instant::now(), tip)
    }
}

impl Drop for Timer {
    fn drop(&mut self) {
        let total_us = self.0.elapsed().as_micros() as u32;
        let ms = total_us / 1000;
        let us = total_us % 1000;
        let tip = self.1;

        info!("{tip}, cost: {ms}.{us:03}ms",);
    }
}
