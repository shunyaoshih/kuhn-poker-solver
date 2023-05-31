use pad::{Alignment, PadStr};
use rand::prelude::*;

use shuffle::irs::Irs;
use shuffle::shuffler::Shuffler;
use std::collections::HashMap;

const NUM_ITERATIONS: u32 = 1_000_000;

// Action:
// - 0 => Bet
// - 1 => Pass
const ACTIONS: [char; 2] = ['b', 'p'];
const NUM_ACTIONS: usize = ACTIONS.len();

// The first 6 information sets are for player 1 and the rests are for player 2.
// Information that is not available to the player who is going to take an action is marked as '.'.
const INFORMATION_SETS: [&str; 12] = [
    "2.", "1.", "0.", "2.pb", "1.pb", "0.pb", ".2b", ".2p", ".1b", ".1p", ".0b", ".0p",
];

struct Trainer {
    pub completed_iterations: u32,
    cards: Vec<usize>,
    information_set_map: HashMap<&'static str, InformationSet>,
}

impl Trainer {
    fn init() -> Self {
        let mut information_set_map = HashMap::new();
        for information_set in INFORMATION_SETS.into_iter() {
            information_set_map.insert(information_set, InformationSet::init());
        }

        Self {
            completed_iterations: 0,
            cards: vec![0, 1, 2],
            information_set_map,
        }
    }

    fn run(&mut self) {
        let mut irs = Irs::default();
        irs.shuffle(&mut self.cards, &mut thread_rng());

        // This essentially covers dealing chance nodes.
        let history: String = self.cards[0].to_string() + &self.cards[1].to_string();

        // For each player, we run CFR algorithm.
        for i in 0..=1 {
            self.cfr(history.as_str(), i, self.completed_iterations, 1.0, 1.0);
        }
        self.completed_iterations += 1;
    }

    fn cfr(
        &mut self,
        history: &str,
        current_player: usize,
        current_iteration: u32,
        player_1_reach_probability: f64,
        player_2_reach_probability: f64,
    ) -> f64 {
        if self.is_terminal(history) {
            return self.utility(history, current_player);
        }

        // Update strategy.
        self.information_set_map
            .get_mut(matched_information_set(history).unwrap())
            .unwrap()
            .compute_strategy();

        // Compute counterfactual values for all actions.
        let mut counterfactual_value = 0.0_f64;
        let mut counterfactual_value_per_action = [0.0_f64; 2];
        for (i, v) in counterfactual_value_per_action.iter_mut().enumerate() {
            let mut new_history: String = history.to_string();
            new_history.push(ACTIONS[i].clone());

            let current_action_probability = self
                .information_set_map
                .get(matched_information_set(history).unwrap())
                .unwrap()
                .strategy[i];
            if self.acting_player(history) == 0 {
                *v = self.cfr(
                    &new_history,
                    current_player,
                    current_iteration,
                    player_1_reach_probability * current_action_probability,
                    player_2_reach_probability,
                )
            } else {
                *v = self.cfr(
                    &new_history,
                    current_player,
                    current_iteration,
                    player_1_reach_probability,
                    player_2_reach_probability * current_action_probability,
                )
            }
            counterfactual_value += current_action_probability * (*v);
        }

        if self.acting_player(history) == current_player {
            let current_player_reach_probability = if current_player == 0 {
                player_1_reach_probability
            } else {
                player_2_reach_probability
            };
            let opponent_player_reach_probability = if current_player == 0 {
                player_2_reach_probability
            } else {
                player_1_reach_probability
            };

            // Accumulate regrets.
            for i in 0..NUM_ACTIONS {
                self.information_set_map
                    .get_mut(matched_information_set(history).unwrap())
                    .unwrap()
                    .cumulative_regrets[i] += opponent_player_reach_probability
                    * (counterfactual_value_per_action[i] - counterfactual_value);
            }

            // Accumulate strategy.
            self.information_set_map
                .get_mut(matched_information_set(history).unwrap())
                .unwrap()
                .add_strategy_to_strategy_sum(current_player_reach_probability);
        }
        counterfactual_value
    }

    fn is_terminal(&self, history: &str) -> bool {
        if history.len() < 4 {
            return false;
        } else if history.len() == 4 {
            return &history[history.len() - 2..] == "pp"
                || &history[history.len() - 2..] == "bb"
                || &history[history.len() - 2..] == "bp";
        }
        &history[history.len() - 3..] == "pbp" || &history[history.len() - 3..] == "pbb"
    }

    fn utility(&self, history: &str, current_player: usize) -> f64 {
        let mut is_player_card_higher: bool =
            history.chars().nth(0).unwrap() > history.chars().nth(1).unwrap();
        if current_player == 1 {
            is_player_card_higher = !is_player_card_higher;
        }

        if &history[history.len() - 2..] == "pp" {
            if is_player_card_higher {
                return 1.0;
            }
            return -1.0;
        } else if &history[history.len() - 2..] == "bb" {
            if is_player_card_higher {
                return 2.0;
            }
            return -2.0;
        } else {
            if history.len() % 2 == 0 && current_player == 0 {
                return 1.0;
            }
            if history.len() % 2 == 1 && current_player == 1 {
                return 1.0;
            }
            return -1.0;
        }
    }

    fn acting_player(&self, history: &str) -> usize {
        history.len() % 2
    }

    fn print_strategy(&self) {
        for information_set in INFORMATION_SETS.into_iter() {
            let s = self
                .information_set_map
                .get(information_set)
                .unwrap()
                .average_strategy();
            println!(
                "{}: [{:.2}, {:.2}]",
                information_set.pad_to_width_with_alignment(5, Alignment::Right),
                s[0],
                s[1]
            );
        }
    }

    fn ev(&self, history: &str, current_player: usize) -> f64 {
        if self.is_terminal(history) {
            return self.utility(history, current_player);
        }

        let strategy = self
            .information_set_map
            .get(matched_information_set(history).unwrap())
            .unwrap()
            .average_strategy();
        let mut ev = 0.0;
        for i in 0..NUM_ACTIONS {
            let mut new_history: String = history.to_string();
            new_history.push(ACTIONS[i].clone());
            ev += strategy[i] * (-self.ev(&new_history, current_player ^ 1));
        }
        ev
    }

    fn print_ev(&self) {
        let mut ev: f64 = 0.0;
        for card1 in 0..=2 {
            for card2 in 0..=2 {
                if card1 == card2 {
                    continue;
                }
                let history: String = card1.to_string() + &card2.to_string();
                ev += self.ev(&history, 0);
            }
        }
        println!("EV for player1: {}", ev / 6.0);
    }
}

fn matched_information_set(history: &str) -> Option<&str> {
    for information_set in INFORMATION_SETS.into_iter() {
        if history.len() != information_set.len() {
            continue;
        }

        let mut matched = true;
        for (a, b) in history.chars().zip(information_set.chars()) {
            if b == '.' {
                continue;
            }
            if a != b {
                matched = false;
                break;
            }
        }
        if matched {
            return Some(information_set);
        }
    }
    None
}

struct InformationSet {
    pub cumulative_regrets: [f64; NUM_ACTIONS],
    pub strategy: [f64; NUM_ACTIONS],
    cumulative_strategy: [f64; NUM_ACTIONS],
}

impl InformationSet {
    fn init() -> Self {
        Self {
            cumulative_regrets: [0.0_f64; NUM_ACTIONS],
            strategy: [0.0_f64; NUM_ACTIONS],
            cumulative_strategy: [0.0_f64; NUM_ACTIONS],
        }
    }

    fn compute_strategy(&mut self) {
        for (s, r) in self.strategy.iter_mut().zip(self.cumulative_regrets) {
            if r > 0.0 {
                *s = r;
            } else {
                *s = 0.0;
            }
        }

        let normalized_sum: f64 = self.strategy.iter().sum();
        if normalized_sum < 1e-6 {
            self.strategy = [0.5; 2];
            return;
        }
        for s in self.strategy.iter_mut() {
            *s /= normalized_sum;
        }
    }

    fn add_strategy_to_strategy_sum(&mut self, realization_weight: f64) {
        for (cs, s) in self.cumulative_strategy.iter_mut().zip(self.strategy) {
            *cs += realization_weight * s;
        }
    }

    fn average_strategy(&self) -> [f64; 2] {
        let normalized_sum: f64 = self.cumulative_strategy.iter().sum();
        if normalized_sum < 1e-6 {
            return [0.0_f64; 2];
        }

        let mut average_strategy = [0.0_f64; 2];
        for (a, s) in average_strategy.iter_mut().zip(self.cumulative_strategy) {
            *a = s / normalized_sum;
        }
        average_strategy
    }
}

fn main() {
    let mut trainer = Trainer::init();
    while trainer.completed_iterations < NUM_ITERATIONS {
        trainer.run();
        if trainer.completed_iterations % 1000 == 0 {
            println!("Iteration {}", trainer.completed_iterations);
            trainer.print_ev();
        }
    }
    trainer.print_strategy();
}
