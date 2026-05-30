/// A room's vibe is ONE number.
pub type Vibe = f64;

/// A JEPA reading — learns to weight prior readings specific to this room.
#[derive(Debug, Clone)]
pub struct Jepa {
    readings: Vec<(f64, f64)>, // (timestamp, value)
    weights: Vec<f64>,         // learned weights for each reading
    window: usize,             // how many readings to consider
}

impl Jepa {
    pub fn new(window: usize) -> Self {
        Self {
            readings: Vec::new(),
            weights: Vec::new(),
            window: window.max(1),
        }
    }

    pub fn read(&mut self, timestamp: f64, value: f64) {
        self.readings.push((timestamp, value));
        // Initialize weight uniformly
        let n = self.readings.len();
        self.weights.push(1.0 / n as f64);
        // Trim to window
        if self.readings.len() > self.window {
            let excess = self.readings.len() - self.window;
            self.readings.drain(..excess);
            self.weights.drain(..excess);
        }
    }

    pub fn predict(&self) -> f64 {
        if self.weights.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = self.weights.iter().sum();
        if total_weight.abs() < 1e-15 {
            return 0.0;
        }
        self.weights
            .iter()
            .zip(self.readings.iter())
            .map(|(w, (_, v))| w * v)
            .sum::<f64>()
            / total_weight
    }

    pub fn surprise(&self, actual: f64) -> f64 {
        (self.predict() - actual).abs()
    }

    pub fn update_weights(&mut self, actual: f64) {
        if self.readings.is_empty() {
            return;
        }
        // Reinforce weights whose readings were close to actual
        for (i, (_, v)) in self.readings.iter().enumerate() {
            let error = (v - actual).abs();
            // Inverse error: closer readings get higher weight
            let score = 1.0 / (1.0 + error);
            self.weights[i] *= score;
        }
        // Normalize
        let sum: f64 = self.weights.iter().sum();
        if sum.abs() > 1e-15 {
            for w in &mut self.weights {
                *w /= sum;
            }
        }
    }
}

/// Murmur — gossip about ONE number.
#[derive(Debug, Clone)]
pub struct Murmur {
    pub source: usize,
    pub vibe: f64,
    pub surprise: f64,
    pub tick: u64,
    pub ttl: u32,
}

impl Murmur {
    pub fn decay(&mut self) {
        if self.ttl > 0 {
            self.ttl -= 1;
        }
    }

    pub fn is_alive(&self) -> bool {
        self.ttl > 0
    }
}

/// Room = vibe + jepa (mono)
#[derive(Debug, Clone)]
pub struct Room {
    pub id: usize,
    pub vibe: Vibe,
    pub jepa: Jepa,
    pub last_surprise: f64,
}

impl Room {
    pub fn new(id: usize, vibe: Vibe, window: usize) -> Self {
        Self {
            id,
            vibe,
            jepa: Jepa::new(window),
            last_surprise: 0.0,
        }
    }

    pub fn tick(&mut self, timestamp: f64, _tick: u64) -> f64 {
        let surprise = self.jepa.surprise(self.vibe);
        self.jepa.read(timestamp, self.vibe);
        self.last_surprise = surprise;
        surprise
    }

    pub fn learn(&mut self) {
        if !self.jepa.readings.is_empty() {
            let last_val = self.jepa.readings.last().map(|(_, v)| *v).unwrap_or(self.vibe);
            self.jepa.update_weights(last_val);
        }
    }
}

/// CellGraph — the full pattern.
#[derive(Debug, Clone)]
pub struct CellGraph {
    pub rooms: Vec<Room>,
    pub edges: Vec<(usize, usize, f64)>, // (from, to, weight)
    pub tick_count: u64,
    pub bpm: f64,
}

impl CellGraph {
    pub fn new(bpm: f64) -> Self {
        Self {
            rooms: Vec::new(),
            edges: Vec::new(),
            tick_count: 0,
            bpm,
        }
    }

    pub fn add_room(&mut self, vibe: Vibe, window: usize) -> usize {
        let id = self.rooms.len();
        self.rooms.push(Room::new(id, vibe, window));
        id
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges.push((from, to, weight));
    }

    pub fn remove_room(&mut self, id: usize) {
        self.rooms.retain(|r| r.id != id);
        self.edges.retain(|(f, t, _)| *f != id && *t != id);
        // Note: we don't re-index; rooms keep their ids
    }

    /// Each room reads its current vibe through its JEPA, gets surprise.
    pub fn tick(&mut self) {
        let timestamp = self.tick_count as f64 / self.bpm * 60.0;
        for room in &mut self.rooms {
            room.tick(timestamp, self.tick_count);
        }
        self.tick_count += 1;
    }

    /// Each room's vibe moves toward weighted average of neighbor vibes.
    pub fn diffuse(&mut self, rate: f64) {
        let n = self.rooms.len();
        if n == 0 {
            return;
        }

        // Compute neighbor influence for each room
        let mut deltas = vec![0.0f64; n];

        for &(from, to, weight) in &self.edges {
            if from < n && to < n {
                let diff = self.rooms[to].vibe - self.rooms[from].vibe;
                deltas[from] += weight * diff;
            }
        }

        // Apply deltas (mutation after reading all)
        for (i, room) in self.rooms.iter_mut().enumerate() {
            room.vibe += rate * deltas[i];
        }
    }

    /// Rooms share their (vibe, surprise) as murmurs.
    pub fn gossip(&self, ttl: u32) -> Vec<Murmur> {
        self.rooms
            .iter()
            .map(|r| Murmur {
                source: r.id,
                vibe: r.vibe,
                surprise: r.last_surprise,
                tick: self.tick_count,
                ttl,
            })
            .collect()
    }

    /// JEPA updates weights based on prediction error.
    pub fn learn(&mut self) {
        for room in &mut self.rooms {
            room.learn();
        }
    }

    /// Total vibe mass.
    pub fn total_vibe(&self) -> f64 {
        self.rooms.iter().map(|r| r.vibe).sum()
    }

    /// Fleet vibe — simple average.
    pub fn fleet_vibe(&self) -> f64 {
        if self.rooms.is_empty() {
            return 0.0;
        }
        self.total_vibe() / self.rooms.len() as f64
    }

    /// Fleet surprise — average surprise.
    pub fn fleet_surprise(&self) -> f64 {
        if self.rooms.is_empty() {
            return 0.0;
        }
        self.rooms.iter().map(|r| r.last_surprise).sum::<f64>() / self.rooms.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. vibe is a single f64
    #[test]
    fn test_vibe_is_single_f64() {
        let v: Vibe = 3.14;
        assert!((v - 3.14).abs() < 1e-10);
    }

    // 2. jepa predict returns weighted average
    #[test]
    fn test_jepa_predict_weighted_average() {
        let mut j = Jepa::new(10);
        j.read(1.0, 2.0);
        j.read(2.0, 4.0);
        let pred = j.predict();
        // Both readings contribute; should be between 2 and 4
        assert!(pred >= 1.5 && pred <= 4.5);
    }

    // 3. jepa surprise is |predicted - actual|
    #[test]
    fn test_jepa_surprise() {
        let mut j = Jepa::new(10);
        j.read(1.0, 10.0);
        let s = j.surprise(10.0);
        assert!(s < 0.01); // predicted ~10, actual 10
    }

    // 4. jepa weights update toward accurate predictors
    #[test]
    fn test_jepa_weights_update() {
        let mut j = Jepa::new(10);
        j.read(1.0, 5.0);
        j.read(2.0, 10.0);
        let w_before = j.weights.clone();
        j.update_weights(10.0);
        // Weight for reading closer to 10 should increase relative
        assert!(j.weights[1] >= w_before[1] * 0.9);
    }

    // 5. jepa learns to weight recent readings more (with constant learning)
    #[test]
    fn test_jepa_weights_recent() {
        let mut j = Jepa::new(10);
        j.read(1.0, 0.0);
        j.read(2.0, 0.0);
        j.read(3.0, 100.0);
        j.update_weights(100.0);
        // The reading at 100 should have higher weight than those at 0
        assert!(j.weights[2] > j.weights[0]);
    }

    // 6. murmur carries one number
    #[test]
    fn test_murmur_one_number() {
        let m = Murmur {
            source: 0,
            vibe: 42.0,
            surprise: 1.0,
            tick: 5,
            ttl: 3,
        };
        assert!((m.vibe - 42.0).abs() < 1e-10);
    }

    // 7. murmur ttl decays
    #[test]
    fn test_murmur_ttl_decay() {
        let mut m = Murmur {
            source: 0,
            vibe: 1.0,
            surprise: 0.0,
            tick: 0,
            ttl: 3,
        };
        m.decay();
        assert_eq!(m.ttl, 2);
        m.decay();
        m.decay();
        assert_eq!(m.ttl, 0);
        assert!(!m.is_alive());
    }

    // 8. room has one vibe
    #[test]
    fn test_room_one_vibe() {
        let r = Room::new(0, 7.5, 5);
        assert!((r.vibe - 7.5).abs() < 1e-10);
    }

    // 9. room jepa is room-specific
    #[test]
    fn test_room_jepa_specific() {
        let mut r1 = Room::new(0, 1.0, 5);
        let mut r2 = Room::new(1, 100.0, 5);
        r1.tick(1.0, 0);
        r1.tick(2.0, 1);
        r2.tick(1.0, 0);
        // r1's jepa has r1's readings, r2's has r2's
        assert_ne!(r1.jepa.predict(), r2.jepa.predict());
    }

    // 10. diffusion converges
    #[test]
    fn test_diffusion_converges() {
        let mut g = CellGraph::new(120.0);
        let _r0 = g.add_room(0.0, 5);
        let _r1 = g.add_room(10.0, 5);
        g.add_edge(r0, r1, 1.0);
        g.add_edge(r1, r0, 1.0);
        for _ in 0..200 {
            g.diffuse(0.1);
        }
        assert!((g.rooms[0].vibe - g.rooms[1].vibe).abs() < 0.1);
    }

    // 11. conservation holds trivially
    #[test]
    fn test_conservation() {
        let mut g = CellGraph::new(120.0);
        let _r0 = g.add_room(3.0, 5);
        let _r1 = g.add_room(7.0, 5);
        g.add_edge(r0, r1, 0.5);
        g.add_edge(r1, r0, 0.5);
        let before = g.total_vibe();
        for _ in 0..100 {
            g.diffuse(0.2);
        }
        let after = g.total_vibe();
        assert!((before - after).abs() < 1e-10);
    }

    // 12. gossip spreads one number
    #[test]
    fn test_gossip() {
        let mut g = CellGraph::new(120.0);
        g.add_room(5.0, 5);
        g.add_room(15.0, 5);
        let murmurs = g.gossip(3);
        assert_eq!(murmurs.len(), 2);
        assert!((murmurs[0].vibe - 5.0).abs() < 1e-10);
        assert!((murmurs[1].vibe - 15.0).abs() < 1e-10);
    }

    // 13. surprise cascades through graph
    #[test]
    fn test_surprise_cascade() {
        let mut g = CellGraph::new(120.0);
        let _r0 = g.add_room(5.0, 5);
        let _r1 = g.add_room(5.0, 5);
        // Let JEPA learn constant 5
        for _ in 0..5 {
            g.tick();
        }
        // Suddenly change room 0
        g.rooms[0].vibe = 100.0;
        g.tick();
        assert!(g.rooms[0].last_surprise > 1.0);
    }

    // 14. different rooms develop different jepa weights
    #[test]
    fn test_different_jepa_weights() {
        let mut g = CellGraph::new(120.0);
        let _r0 = g.add_room(1.0, 5);
        let _r1 = g.add_room(100.0, 5);
        for i in 0..10 {
            g.rooms[0].vibe = (i as f64).sin();
            g.rooms[1].vibe = (i as f64).cos() * 10.0;
            g.tick();
            g.learn();
        }
        let w0 = g.rooms[0].jepa.weights.clone();
        let w1 = g.rooms[1].jepa.weights.clone();
        // They should differ
        let diff: f64 = w0.iter().zip(w1.iter()).map(|(a, b)| (a - b).abs()).sum();
        assert!(diff > 0.01);
    }

    // 15. constant values → near-zero surprise
    #[test]
    fn test_constant_near_zero_surprise() {
        let mut g = CellGraph::new(120.0);
        g.add_room(5.0, 10);
        for _ in 0..10 {
            g.tick();
        }
        assert!(g.rooms[0].last_surprise < 0.01);
    }

    // 16. oscillating values → higher surprise
    #[test]
    fn test_oscillating_higher_surprise() {
        let mut g = CellGraph::new(120.0);
        g.add_room(0.0, 10);
        for i in 0..20 {
            g.rooms[0].vibe = if i % 2 == 0 { 0.0 } else { 100.0 };
            g.tick();
        }
        // After oscillation, surprise should be significant
        assert!(g.rooms[0].last_surprise > 1.0);
    }

    // 17. removing a room preserves conservation
    #[test]
    fn test_remove_preserves_conservation() {
        let mut g = CellGraph::new(120.0);
        let _r0 = g.add_room(3.0, 5);
        let _r1 = g.add_room(7.0, 5);
        let _r2 = g.add_room(10.0, 5);
        let before = g.total_vibe();
        g.remove_room(r1);
        // Total should be reduced by room 1's vibe
        assert!((g.total_vibe() - (before - 7.0)).abs() < 1e-10);
    }

    // 18. adding a room adjusts total mass
    #[test]
    fn test_add_adjusts_mass() {
        let mut g = CellGraph::new(120.0);
        g.add_room(3.0, 5);
        let before = g.total_vibe();
        g.add_room(7.0, 5);
        assert!((g.total_vibe() - before - 7.0).abs() < 1e-10);
    }

    // 19. mono diffusion is fast (perf sanity)
    #[test]
    fn test_mono_diffusion_speed() {
        let mut g = CellGraph::new(120.0);
        for i in 0..100 {
            g.add_room(i as f64, 5);
        }
        for i in 0..100 {
            for j in 0..100 {
                if i != j {
                    g.add_edge(i, j, 0.01);
                }
            }
        }
        let start = std::time::Instant::now();
        for _ in 0..1000 {
            g.diffuse(0.01);
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_millis() < 5000); // should be way under 5s
    }

    // 20. jepa with window=1 is just last reading
    #[test]
    fn test_jepa_window_one() {
        let mut j = Jepa::new(1);
        j.read(1.0, 10.0);
        j.read(2.0, 20.0);
        let pred = j.predict();
        assert!((pred - 20.0).abs() < 1e-10);
    }

    // 21. jepa with large window smooths
    #[test]
    fn test_jepa_large_window_smooths() {
        let mut j = Jepa::new(100);
        for i in 0..50 {
            j.read(i as f64, 50.0 + (i as f64 * 0.1).sin() * 10.0);
        }
        let pred = j.predict();
        // Should be near 50 (the center)
        assert!(pred > 45.0 && pred < 55.0);
    }

    // 22. edge weight affects diffusion speed
    #[test]
    fn test_edge_weight_affects_diffusion() {
        let mut g1 = CellGraph::new(120.0);
        let r0 = g1.add_room(0.0, 5);
        let r1 = g1.add_room(10.0, 5);
        g1.add_edge(r0, r1, 0.1);
        g1.add_edge(r1, r0, 0.1);

        let mut g2 = CellGraph::new(120.0);
        let r0b = g2.add_room(0.0, 5);
        let r1b = g2.add_room(10.0, 5);
        g2.add_edge(r0b, r1b, 1.0);
        g2.add_edge(r1b, r0b, 1.0);

        for _ in 0..10 {
            g1.diffuse(0.1);
            g2.diffuse(0.1);
        }
        let diff1 = (g1.rooms[0].vibe - g1.rooms[1].vibe).abs();
        let diff2 = (g2.rooms[0].vibe - g2.rooms[1].vibe).abs();
        assert!(diff2 < diff1); // stronger edge → closer
    }

    // 23. tick increments correctly
    #[test]
    fn test_tick_increments() {
        let mut g = CellGraph::new(120.0);
        g.add_room(5.0, 5);
        assert_eq!(g.tick_count, 0);
        g.tick();
        assert_eq!(g.tick_count, 1);
        g.tick();
        assert_eq!(g.tick_count, 2);
    }

    // 24. empty graph handles gracefully
    #[test]
    fn test_empty_graph() {
        let mut g = CellGraph::new(120.0);
        g.tick();
        g.diffuse(0.1);
        g.learn();
        let murmurs = g.gossip(3);
        assert!(murmurs.is_empty());
        assert_eq!(g.fleet_vibe(), 0.0);
        assert_eq!(g.fleet_surprise(), 0.0);
    }

    // 25. single room doesn't crash
    #[test]
    fn test_single_room() {
        let mut g = CellGraph::new(120.0);
        g.add_room(5.0, 5);
        for _ in 0..10 {
            g.tick();
            g.diffuse(0.1);
            g.learn();
        }
        assert!((g.rooms[0].vibe - 5.0).abs() < 1e-10);
    }

    // 26. fleet vibe is simple average
    #[test]
    fn test_fleet_vibe_average() {
        let mut g = CellGraph::new(120.0);
        g.add_room(0.0, 5);
        g.add_room(10.0, 5);
        g.add_room(20.0, 5);
        assert!((g.fleet_vibe() - 10.0).abs() < 1e-10);
    }

    // 27. fleet surprise is average surprise
    #[test]
    fn test_fleet_surprise_average() {
        let mut g = CellGraph::new(120.0);
        g.add_room(5.0, 10);
        g.add_room(5.0, 10);
        for _ in 0..5 {
            g.tick();
        }
        // Constant input → near zero surprise → fleet surprise near zero
        assert!(g.fleet_surprise() < 0.1);
    }
}
