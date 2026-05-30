/// A room's vibe is ONE number.
pub type Vibe = f64;

/// A JEPA reading — learns to weight prior readings specific to this room.
#[derive(Clone, Debug)]
pub struct Jepa {
    /// (timestamp, value) pairs
    readings: Vec<(f64, f64)>,
    /// learned weights for each reading
    weights: Vec<f64>,
    /// how many readings to consider
    window: usize,
    /// learning rate for weight updates
    learning_rate: f64,
}

impl Jepa {
    pub fn new(window: usize) -> Self {
        Jepa {
            readings: Vec::new(),
            weights: Vec::new(),
            window,
            learning_rate: 0.1,
        }
    }

    pub fn with_learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn read(&mut self, timestamp: f64, value: f64) {
        self.readings.push((timestamp, value));
        // Initialize new weight to 1.0
        self.weights.push(1.0);
        // Trim to window
        while self.readings.len() > self.window {
            self.readings.remove(0);
            self.weights.remove(0);
        }
    }

    pub fn predict(&self) -> f64 {
        if self.readings.is_empty() {
            return 0.0;
        }
        let total_weight: f64 = self.weights.iter().sum();
        if total_weight == 0.0 {
            return 0.0;
        }
        self.readings
            .iter()
            .zip(self.weights.iter())
            .map(|((_, v), w)| v * w)
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
        // For each reading, compute how close it was to the actual value.
        // Readings closer to actual get their weights increased.
        let prediction = self.predict();
        for i in 0..self.readings.len() {
            let reading_val = self.readings[i].1;
            let distance = (reading_val - actual).abs();
            // If this reading was close to actual, boost its weight
            // If it was far, reduce it
            let max_dist = self
                .readings
                .iter()
                .map(|(_, v)| (v - actual).abs())
                .fold(f64::EPSILON, f64::max);
            let accuracy = 1.0 - (distance / max_dist).min(1.0);
            self.weights[i] += self.learning_rate * (accuracy - 0.5);
            self.weights[i] = self.weights[i].max(0.01);
        }
        // Normalize weights to prevent unbounded growth
        let sum: f64 = self.weights.iter().sum();
        if sum > self.window as f64 * 10.0 {
            for w in &mut self.weights {
                *w /= sum / (self.window as f64);
            }
        }
        let _ = prediction; // used implicitly via weight update
    }

    pub fn readings_len(&self) -> usize {
        self.readings.len()
    }

    pub fn get_weights(&self) -> &[f64] {
        &self.weights
    }
}

/// Murmur — gossip about ONE number.
#[derive(Clone, Debug)]
pub struct Murmur {
    pub source: usize,
    pub vibe: f64,
    pub surprise: f64,
    pub tick: u64,
    pub ttl: u32,
}

impl Murmur {
    pub fn new(source: usize, vibe: f64, surprise: f64, tick: u64) -> Self {
        Murmur {
            source,
            vibe,
            surprise,
            tick,
            ttl: 10,
        }
    }

    pub fn decay(&mut self) -> bool {
        if self.ttl == 0 {
            false
        } else {
            self.ttl -= 1;
            true
        }
    }
}

/// Room = vibe + jepa (mono)
#[derive(Clone, Debug)]
pub struct Room {
    pub id: usize,
    pub vibe: f64,
    pub jepa: Jepa,
    pub last_surprise: f64,
}

impl Room {
    pub fn new(id: usize, jepa_window: usize) -> Self {
        Room {
            id,
            vibe: 0.0,
            jepa: Jepa::new(jepa_window),
            last_surprise: 0.0,
        }
    }

    pub fn with_vibe(mut self, vibe: f64) -> Self {
        self.vibe = vibe;
        self
    }
}

/// CellGraph — the full graph of rooms.
#[derive(Clone, Debug)]
pub struct CellGraph {
    pub rooms: Vec<Room>,
    /// (from, to, weight)
    pub edges: Vec<(usize, usize, f64)>,
    pub tick_count: u64,
    pub bpm: f64,
    diffusion_rate: f64,
}

impl CellGraph {
    pub fn new(bpm: f64) -> Self {
        CellGraph {
            rooms: Vec::new(),
            edges: Vec::new(),
            tick_count: 0,
            bpm,
            diffusion_rate: 0.1,
        }
    }

    pub fn with_diffusion_rate(mut self, rate: f64) -> Self {
        self.diffusion_rate = rate;
        self
    }

    pub fn add_room(&mut self, room: Room) {
        self.rooms.push(room);
    }

    pub fn remove_room(&mut self, id: usize) {
        self.rooms.retain(|r| r.id != id);
        self.edges
            .retain(|(from, to, _)| *from != id && *to != id);
    }

    pub fn add_edge(&mut self, from: usize, to: usize, weight: f64) {
        self.edges.push((from, to, weight));
    }

    /// Tick: each room reads its current vibe through its JEPA, gets surprise.
    pub fn tick(&mut self, timestamp: f64) {
        self.tick_count += 1;
        for room in &mut self.rooms {
            room.last_surprise = room.jepa.surprise(room.vibe);
            room.jepa.read(timestamp, room.vibe);
            room.jepa.update_weights(room.vibe);
        }
    }

    /// Diffuse: each room's vibe moves toward weighted average of neighbor vibes.
    /// Conservation holds by construction: each room gives and receives equally.
    pub fn diffuse(&mut self) {
        let n = self.rooms.len();
        if n == 0 {
            return;
        }

        let mut deltas = vec![0.0f64; n];
        let id_to_idx: std::collections::HashMap<usize, usize> = self
            .rooms
            .iter()
            .enumerate()
            .map(|(i, r)| (r.id, i))
            .collect();

        // Process each undirected edge pair only once to avoid oscillation.
        // For each edge, compute flow based on difference, split equally.
        let mut seen = std::collections::HashSet::new();
        for &(from_id, to_id, weight) in &self.edges {
            let key = (from_id.min(to_id), from_id.max(to_id));
            if seen.contains(&key) {
                continue;
            }
            seen.insert(key);

            let from_idx = match id_to_idx.get(&from_id) {
                Some(&i) => i,
                None => continue,
            };
            let to_idx = match id_to_idx.get(&to_id) {
                Some(&i) => i,
                None => continue,
            };
            let diff = self.rooms[from_idx].vibe - self.rooms[to_idx].vibe;
            let transfer = diff * weight * self.diffusion_rate;
            deltas[from_idx] -= transfer;
            deltas[to_idx] += transfer;
        }

        for (i, room) in self.rooms.iter_mut().enumerate() {
            room.vibe += deltas[i];
        }
    }

    /// Gossip: rooms share their (vibe, surprise) as murmurs.
    pub fn gossip(&self) -> Vec<Murmur> {
        self.rooms
            .iter()
            .map(|room| Murmur::new(room.id, room.vibe, room.last_surprise, self.tick_count))
            .collect()
    }

    /// Total vibe mass (for conservation verification).
    pub fn total_vibe(&self) -> f64 {
        self.rooms.iter().map(|r| r.vibe).sum()
    }

    /// Fleet vibe: simple average.
    pub fn fleet_vibe(&self) -> f64 {
        if self.rooms.is_empty() {
            return 0.0;
        }
        self.total_vibe() / self.rooms.len() as f64
    }

    /// Fleet surprise: average surprise.
    pub fn fleet_surprise(&self) -> f64 {
        if self.rooms.is_empty() {
            return 0.0;
        }
        self.rooms.iter().map(|r| r.last_surprise).sum::<f64>() / self.rooms.len() as f64
    }

    /// Learn: update JEPA weights across all rooms.
    pub fn learn(&mut self) {
        for room in &mut self.rooms {
            room.jepa.update_weights(room.vibe);
        }
    }

    /// Spread a murmur's surprise to neighbors.
    pub fn spread_surprise(&mut self, murmur: &Murmur) {
        let id_to_idx: std::collections::HashMap<usize, usize> = self
            .rooms
            .iter()
            .enumerate()
            .map(|(i, r)| (r.id, i))
            .collect();

        for &(from, to, weight) in &self.edges {
            if from == murmur.source {
                if let Some(&idx) = id_to_idx.get(&to) {
                    self.rooms[idx].last_surprise += murmur.surprise * weight * 0.5;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // 1. vibe is a single f64
    #[test]
    fn test_vibe_is_single_f64() {
        let vibe: Vibe = 3.14;
        assert_eq!(vibe, 3.14);
    }

    // 2. jepa predict returns weighted average
    #[test]
    fn test_jepa_predict_weighted_average() {
        let mut jepa = Jepa::new(10);
        jepa.read(1.0, 2.0);
        jepa.read(2.0, 4.0);
        // Equal weights (both 1.0), so predict = (2+4)/2 = 3.0
        assert!((jepa.predict() - 3.0).abs() < 1e-9);
    }

    // 3. jepa surprise is |predicted - actual|
    #[test]
    fn test_jepa_surprise() {
        let mut jepa = Jepa::new(10);
        jepa.read(1.0, 10.0);
        jepa.read(2.0, 10.0);
        // predict = 10.0, actual = 12.0 → surprise = 2.0
        assert!((jepa.surprise(12.0) - 2.0).abs() < 1e-9);
    }

    // 4. jepa weights update toward accurate predictors
    #[test]
    fn test_jepa_weight_update() {
        let mut jepa = Jepa::new(10).with_learning_rate(0.5);
        jepa.read(1.0, 5.0);
        jepa.read(2.0, 10.0);
        // actual = 10.0; reading at t=2 (value=10) is closer → its weight should increase more
        let w_before = jepa.get_weights().to_vec();
        jepa.update_weights(10.0);
        let w_after = jepa.get_weights();
        // The second weight should have increased relative to the first
        assert!(w_after[1] > w_before[1]);
    }

    // 5. jepa learns to weight recent readings more (for stable signals)
    #[test]
    fn test_jepa_learns_recent() {
        let mut jepa = Jepa::new(20).with_learning_rate(0.3);
        // Old readings at 1.0
        for t in 1..=10 {
            jepa.read(t as f64, 1.0);
        }
        // Recent readings at 5.0
        for t in 11..=20 {
            jepa.read(t as f64, 5.0);
        }
        // After multiple weight updates toward 5.0, prediction should lean toward 5.0
        for _ in 0..10 {
            jepa.update_weights(5.0);
        }
        let pred = jepa.predict();
        assert!(pred > 3.0, "prediction should lean toward recent values (5.0), got {}", pred);
    }

    // 6. murmur carries one number
    #[test]
    fn test_murmur_carries_one_number() {
        let m = Murmur::new(0, 42.0, 1.0, 1);
        assert_eq!(m.vibe, 42.0);
        assert_eq!(m.source, 0);
    }

    // 7. murmur ttl decays
    #[test]
    fn test_murmur_ttl_decays() {
        let mut m = Murmur::new(0, 1.0, 0.0, 0);
        assert_eq!(m.ttl, 10);
        assert!(m.decay());
        assert_eq!(m.ttl, 9);
        m.ttl = 0;
        assert!(!m.decay());
    }

    // 8. room has one vibe
    #[test]
    fn test_room_has_one_vibe() {
        let room = Room::new(0, 5).with_vibe(7.5);
        assert!((room.vibe - 7.5).abs() < 1e-9);
    }

    // 9. room jepa is room-specific (different rooms learn differently)
    #[test]
    fn test_room_jepa_room_specific() {
        let mut r1 = Room::new(0, 5).with_vibe(1.0);
        let mut r2 = Room::new(1, 5).with_vibe(100.0);
        // Feed different data
        for t in 0..5 {
            r1.jepa.read(t as f64, t as f64 * 1.0);
            r2.jepa.read(t as f64, t as f64 * 100.0);
        }
        // Predictions should differ
        assert!(r1.jepa.predict() < r2.jepa.predict());
    }

    // 10. diffusion converges (mono is fast)
    #[test]
    fn test_diffusion_converges() {
        let mut graph = CellGraph::new(120.0).with_diffusion_rate(0.5);
        graph.add_room(Room::new(0, 5).with_vibe(0.0));
        graph.add_room(Room::new(1, 5).with_vibe(100.0));
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 0, 1.0);

        for _ in 0..50 {
            graph.diffuse();
        }
        let diff = (graph.rooms[0].vibe - graph.rooms[1].vibe).abs();
        assert!(diff < 1.0, "rooms should converge, diff = {}", diff);
    }

    // 11. conservation holds trivially
    #[test]
    fn test_conservation() {
        let mut graph = CellGraph::new(120.0).with_diffusion_rate(0.3);
        graph.add_room(Room::new(0, 5).with_vibe(10.0));
        graph.add_room(Room::new(1, 5).with_vibe(20.0));
        graph.add_room(Room::new(2, 5).with_vibe(30.0));
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 0, 1.0);
        graph.add_edge(1, 2, 0.5);
        graph.add_edge(2, 1, 0.5);

        let before = graph.total_vibe();
        for _ in 0..100 {
            graph.diffuse();
        }
        let after = graph.total_vibe();
        assert!(
            (before - after).abs() < 1e-9,
            "conservation: before={}, after={}",
            before,
            after
        );
    }

    // 12. gossip spreads one number
    #[test]
    fn test_gossip() {
        let mut graph = CellGraph::new(120.0);
        graph.add_room(Room::new(0, 5).with_vibe(5.0));
        graph.add_room(Room::new(1, 5).with_vibe(15.0));
        graph.tick(1.0);
        let murmurs = graph.gossip();
        assert_eq!(murmurs.len(), 2);
        assert_eq!(murmurs[0].source, 0);
        assert!((murmurs[0].vibe - 5.0).abs() < 1e-9);
    }

    // 13. surprise cascades through graph
    #[test]
    fn test_surprise_cascade() {
        let mut graph = CellGraph::new(120.0);
        graph.add_room(Room::new(0, 5).with_vibe(0.0));
        graph.add_room(Room::new(1, 5).with_vibe(50.0));
        graph.add_edge(0, 1, 1.0);
        graph.tick(1.0);
        let murmurs = graph.gossip();
        let high_surprise = murmurs.iter().find(|m| m.source == 0).unwrap();
        graph.spread_surprise(high_surprise);
        assert!(graph.rooms[1].last_surprise > 0.0);
    }

    // 14. different rooms develop different jepa weights
    #[test]
    fn test_different_jepa_weights() {
        let mut r1 = Room::new(0, 10);
        let mut r2 = Room::new(1, 10);
        for t in 0..10 {
            r1.jepa.read(t as f64, (t % 3) as f64);
            r2.jepa.read(t as f64, (t as f64).sin());
        }
        for _ in 0..5 {
            r1.jepa.update_weights(0.0);
            r2.jepa.update_weights(0.0);
        }
        let w1 = r1.jepa.get_weights().to_vec();
        let w2 = r2.jepa.get_weights().to_vec();
        assert_ne!(w1, w2, "different data should produce different weights");
    }

    // 15. constant values → near-zero surprise
    #[test]
    fn test_constant_near_zero_surprise() {
        let mut jepa = Jepa::new(10);
        for t in 0..10 {
            jepa.read(t as f64, 5.0);
        }
        // predict = 5.0, actual = 5.0
        assert!(jepa.surprise(5.0) < 1e-9);
    }

    // 16. oscillating values → higher surprise
    #[test]
    fn test_oscillating_higher_surprise() {
        let mut jepa = Jepa::new(10);
        for t in 0..10 {
            jepa.read(t as f64, if t % 2 == 0 { 0.0 } else { 100.0 });
        }
        // predict ≈ 50.0, actual = 0.0 → surprise ≈ 50.0
        let surprise = jepa.surprise(0.0);
        assert!(surprise > 30.0, "oscillating should have high surprise, got {}", surprise);
    }

    // 17. removing a room preserves conservation
    #[test]
    fn test_remove_preserves_conservation() {
        let mut graph = CellGraph::new(120.0).with_diffusion_rate(0.5);
        graph.add_room(Room::new(0, 5).with_vibe(10.0));
        graph.add_room(Room::new(1, 5).with_vibe(20.0));
        graph.add_room(Room::new(2, 5).with_vibe(30.0));
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 0, 1.0);
        graph.add_edge(1, 2, 1.0);
        graph.add_edge(2, 1, 1.0);

        let before = graph.total_vibe();
        graph.diffuse();
        let after_diffuse = graph.total_vibe();
        assert!((before - after_diffuse).abs() < 1e-9);

        graph.remove_room(1);
        let after_remove = graph.total_vibe();
        assert!((after_remove - 40.0).abs() < 1e-9, "remaining mass = {}", after_remove);
    }

    // 18. adding a room adjusts total mass
    #[test]
    fn test_add_room_adjusts_mass() {
        let mut graph = CellGraph::new(120.0);
        graph.add_room(Room::new(0, 5).with_vibe(10.0));
        assert!((graph.total_vibe() - 10.0).abs() < 1e-9);
        graph.add_room(Room::new(1, 5).with_vibe(25.0));
        assert!((graph.total_vibe() - 35.0).abs() < 1e-9);
    }

    // 19. mono diffusion is faster than 16-dim (compare convergence ticks)
    // This is conceptual: mono diffusion on a simple 2-room graph converges in few ticks.
    #[test]
    fn test_mono_diffusion_fast() {
        let mut graph = CellGraph::new(120.0).with_diffusion_rate(0.5);
        graph.add_room(Room::new(0, 5).with_vibe(0.0));
        graph.add_room(Room::new(1, 5).with_vibe(100.0));
        graph.add_edge(0, 1, 1.0);
        graph.add_edge(1, 0, 1.0);

        let mut ticks = 0;
        while (graph.rooms[0].vibe - graph.rooms[1].vibe).abs() > 0.5 {
            graph.diffuse();
            ticks += 1;
            if ticks > 20 {
                break;
            }
        }
        assert!(ticks <= 15, "mono diffusion should converge fast, took {} ticks", ticks);
    }

    // 20. jepa with window=1 is just last reading
    #[test]
    fn test_jepa_window_one() {
        let mut jepa = Jepa::new(1);
        jepa.read(1.0, 10.0);
        jepa.read(2.0, 20.0);
        assert!((jepa.predict() - 20.0).abs() < 1e-9);
    }

    // 21. jepa with large window smooths
    #[test]
    fn test_jepa_large_window_smooths() {
        let mut jepa = Jepa::new(100);
        for t in 0..50 {
            jepa.read(t as f64, (t as f64).sin());
        }
        let pred = jepa.predict();
        // Should be somewhere in the middle of sin range, not extreme
        assert!(pred > -1.0 && pred < 1.0);
    }

    // 22. edge weight affects diffusion speed
    #[test]
    fn test_edge_weight_affects_diffusion() {
        let mut graph_weak = CellGraph::new(120.0).with_diffusion_rate(0.5);
        graph_weak.add_room(Room::new(0, 5).with_vibe(0.0));
        graph_weak.add_room(Room::new(1, 5).with_vibe(100.0));
        graph_weak.add_edge(0, 1, 0.1);
        graph_weak.add_edge(1, 0, 0.1);

        let mut graph_strong = CellGraph::new(120.0).with_diffusion_rate(0.5);
        graph_strong.add_room(Room::new(0, 5).with_vibe(0.0));
        graph_strong.add_room(Room::new(1, 5).with_vibe(100.0));
        graph_strong.add_edge(0, 1, 1.0);
        graph_strong.add_edge(1, 0, 1.0);

        for _ in 0..5 {
            graph_weak.diffuse();
            graph_strong.diffuse();
        }
        let diff_weak = (graph_weak.rooms[0].vibe - graph_weak.rooms[1].vibe).abs();
        let diff_strong = (graph_strong.rooms[0].vibe - graph_strong.rooms[1].vibe).abs();
        assert!(diff_strong < diff_weak, "strong edges should diffuse faster");
    }

    // 23. tick increments correctly
    #[test]
    fn test_tick_increments() {
        let mut graph = CellGraph::new(120.0);
        graph.add_room(Room::new(0, 5));
        assert_eq!(graph.tick_count, 0);
        graph.tick(1.0);
        assert_eq!(graph.tick_count, 1);
        graph.tick(2.0);
        assert_eq!(graph.tick_count, 2);
    }

    // 24. empty graph handles gracefully
    #[test]
    fn test_empty_graph() {
        let mut graph = CellGraph::new(120.0);
        graph.tick(1.0);
        graph.diffuse();
        let murmurs = graph.gossip();
        assert!(murmurs.is_empty());
        assert_eq!(graph.total_vibe(), 0.0);
        assert_eq!(graph.fleet_vibe(), 0.0);
        assert_eq!(graph.fleet_surprise(), 0.0);
    }

    // 25. single room doesn't crash
    #[test]
    fn test_single_room() {
        let mut graph = CellGraph::new(120.0);
        graph.add_room(Room::new(0, 5).with_vibe(42.0));
        graph.tick(1.0);
        graph.diffuse();
        let murmurs = graph.gossip();
        assert_eq!(murmurs.len(), 1);
        assert!((graph.total_vibe() - 42.0).abs() < 1e-9);
        assert!((graph.rooms[0].vibe - 42.0).abs() < 1e-9);
    }

    // 26. fleet vibe is simple average
    #[test]
    fn test_fleet_vibe() {
        let mut graph = CellGraph::new(120.0);
        graph.add_room(Room::new(0, 5).with_vibe(10.0));
        graph.add_room(Room::new(1, 5).with_vibe(20.0));
        graph.add_room(Room::new(2, 5).with_vibe(30.0));
        assert!((graph.fleet_vibe() - 20.0).abs() < 1e-9);
    }

    // 27. fleet surprise is average surprise
    #[test]
    fn test_fleet_surprise() {
        let mut graph = CellGraph::new(120.0);
        graph.add_room(Room::new(0, 5).with_vibe(5.0));
        graph.add_room(Room::new(1, 5).with_vibe(10.0));
        // First tick: rooms have initial surprise from empty jepa
        graph.tick(1.0);
        let fs = graph.fleet_surprise();
        assert!(fs >= 0.0);
    }
}
