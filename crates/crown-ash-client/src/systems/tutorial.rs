//! First-run tutorial overlay for Crown & Ash.
//!
//! The tutorial is intentionally lightweight and client-only: it guides a new
//! player through the UI affordances that already exist (selection, detail
//! panels, actions, diplomacy, and event feed) without blocking observer mode or
//! requiring server-side state.

use bevy::input::ButtonInput;
use bevy::prelude::*;
use bevy_egui::{egui, EguiContexts};

use crate::resources::game_state::ClientGameState;
use crate::resources::selection::Selection;

/// One step in the first-run tutorial.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TutorialStep {
    Welcome,
    SelectProvince,
    ReadDetails,
    SelectFaction,
    UseActions,
    FollowChronicle,
    Complete,
}

impl TutorialStep {
    const ORDER: [TutorialStep; 7] = [
        TutorialStep::Welcome,
        TutorialStep::SelectProvince,
        TutorialStep::ReadDetails,
        TutorialStep::SelectFaction,
        TutorialStep::UseActions,
        TutorialStep::FollowChronicle,
        TutorialStep::Complete,
    ];

    fn index(self) -> usize {
        Self::ORDER
            .iter()
            .position(|step| *step == self)
            .unwrap_or(0)
    }

    fn next(self) -> TutorialStep {
        let idx = self.index();
        Self::ORDER
            .get(idx + 1)
            .copied()
            .unwrap_or(TutorialStep::Complete)
    }

    fn previous(self) -> TutorialStep {
        let idx = self.index();
        idx.checked_sub(1)
            .and_then(|prev| Self::ORDER.get(prev))
            .copied()
            .unwrap_or(TutorialStep::Welcome)
    }

    fn title(self) -> &'static str {
        match self {
            TutorialStep::Welcome => "Welcome to Crown & Ash",
            TutorialStep::SelectProvince => "1. Select a province",
            TutorialStep::ReadDetails => "2. Read the detail panel",
            TutorialStep::SelectFaction => "3. Inspect a faction",
            TutorialStep::UseActions => "4. Try realm actions",
            TutorialStep::FollowChronicle => "5. Follow the chronicle",
            TutorialStep::Complete => "Tutorial complete",
        }
    }

    fn body(self) -> &'static str {
        match self {
            TutorialStep::Welcome => {
                "This quick tour highlights the main controls. You can continue as an observer, join with a wallet, or hide this guide at any time."
            }
            TutorialStep::SelectProvince => {
                "Click a province on the 3D map or on the minimap. Province selection drives the right-side detail panel and unlocks local actions."
            }
            TutorialStep::ReadDetails => {
                "The detail panel shows population, terrain, resources, improvements, armies, faith, trade, sieges, and generated narrative history."
            }
            TutorialStep::SelectFaction => {
                "Use the Factions panel, Tab, or number keys 1-7 to inspect realms. Selecting a faction also jumps to one of its provinces."
            }
            TutorialStep::UseActions => {
                "The Actions window appears near the lower-left. With a province selected, you can raise armies, build improvements, set taxes, trade, convert faith, and command armies."
            }
            TutorialStep::FollowChronicle => {
                "The bottom chronicle groups events by turn and highlights important wars, battles, plagues, births, deaths, and diplomacy. Toasts surface the biggest changes."
            }
            TutorialStep::Complete => {
                "You are ready to play. Press F1 to reopen this tutorial, Esc to clear selection, Tab to cycle factions, and the mouse wheel to zoom."
            }
        }
    }

    fn completion_hint(self) -> &'static str {
        match self {
            TutorialStep::Welcome => "Click Next to start the guided tour.",
            TutorialStep::SelectProvince => "Goal: select any province.",
            TutorialStep::ReadDetails => "Goal: keep a province selected and review its details.",
            TutorialStep::SelectFaction => {
                "Goal: select any faction from the left panel or keyboard."
            }
            TutorialStep::UseActions => {
                "Goal: select a province so the Actions window shows province actions."
            }
            TutorialStep::FollowChronicle => {
                "Goal: wait for events, or press Next if you are observing an idle realm."
            }
            TutorialStep::Complete => "Press Done to hide this overlay.",
        }
    }
}

/// Client-only tutorial state.
#[derive(Resource, Debug, Clone)]
pub struct TutorialState {
    /// Whether the overlay is currently visible.
    pub visible: bool,
    /// Current step in the tutorial sequence.
    pub step: TutorialStep,
    /// True once the user has reached or skipped the final step.
    pub completed: bool,
    /// Auto-advance is disabled after a manual Back click so users can review.
    pub auto_advance: bool,
}

impl Default for TutorialState {
    fn default() -> Self {
        Self {
            visible: true,
            step: TutorialStep::Welcome,
            completed: false,
            auto_advance: true,
        }
    }
}

impl TutorialState {
    fn advance(&mut self) {
        self.step = self.step.next();
        self.completed = self.step == TutorialStep::Complete;
        self.auto_advance = true;
    }

    fn back(&mut self) {
        self.step = self.step.previous();
        self.completed = false;
        self.auto_advance = false;
    }

    fn skip(&mut self) {
        self.step = TutorialStep::Complete;
        self.completed = true;
        self.visible = false;
        self.auto_advance = false;
    }

    fn reset(&mut self) {
        *self = Self::default();
    }
}

/// Toggle the tutorial with F1 and reset it with Shift+F1.
pub fn tutorial_shortcuts(keys: Res<ButtonInput<KeyCode>>, mut tutorial: ResMut<TutorialState>) {
    if keys.just_pressed(KeyCode::F1) {
        if keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]) {
            tutorial.reset();
        } else {
            tutorial.visible = !tutorial.visible;
        }
    }
}

/// Render the tutorial overlay and auto-advance steps as the player completes
/// their goals.
pub fn tutorial_overlay(
    mut contexts: EguiContexts,
    mut tutorial: ResMut<TutorialState>,
    selection: Res<Selection>,
    game_state: Res<ClientGameState>,
) {
    if !tutorial.visible {
        return;
    }

    if tutorial.auto_advance && step_goal_met(tutorial.step, &selection, &game_state) {
        tutorial.advance();
    }

    let ctx = contexts.ctx_mut();
    let progress = tutorial.step.index() + 1;
    let total = TutorialStep::ORDER.len();
    let title = tutorial.step.title();
    let body = tutorial.step.body();
    let hint = tutorial.step.completion_hint();
    let can_go_back = tutorial.step != TutorialStep::Welcome;
    let is_complete = tutorial.step == TutorialStep::Complete;

    let mut clicked_back = false;
    let mut clicked_next = false;
    let mut clicked_skip = false;
    let mut clicked_done = false;

    egui::Window::new(title)
        .anchor(egui::Align2::CENTER_TOP, [0.0, 58.0])
        .default_width(430.0)
        .resizable(false)
        .collapsible(true)
        .show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(format!("Step {progress}/{total}"));
                ui.add(egui::ProgressBar::new(progress as f32 / total as f32).desired_width(220.0));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(egui::RichText::new("F1").monospace().small());
                });
            });

            ui.separator();
            ui.label(body);
            ui.add_space(6.0);
            ui.colored_label(
                egui::Color32::from_rgb(180, 200, 255),
                egui::RichText::new(hint).italics(),
            );

            if let Some(context) = current_context(&selection, &game_state) {
                ui.add_space(6.0);
                ui.label(
                    egui::RichText::new(context)
                        .small()
                        .color(egui::Color32::GRAY),
                );
            }

            ui.add_space(10.0);
            ui.horizontal(|ui| {
                if ui
                    .add_enabled(can_go_back, egui::Button::new("Back"))
                    .clicked()
                {
                    clicked_back = true;
                }

                if is_complete {
                    if ui.button("Done").clicked() {
                        clicked_done = true;
                    }
                } else if ui.button("Next").clicked() {
                    clicked_next = true;
                }

                if ui.button("Skip").clicked() {
                    clicked_skip = true;
                }

                if ui.button("Reset").clicked() {
                    tutorial.reset();
                }
            });
        });

    if clicked_back {
        tutorial.back();
    }
    if clicked_next {
        tutorial.advance();
    }
    if clicked_skip {
        tutorial.skip();
    }
    if clicked_done {
        tutorial.visible = false;
        tutorial.completed = true;
    }
}

fn step_goal_met(step: TutorialStep, selection: &Selection, game_state: &ClientGameState) -> bool {
    match step {
        TutorialStep::Welcome => false,
        TutorialStep::SelectProvince => selection.province.is_some(),
        TutorialStep::ReadDetails => selection.province.is_some(),
        TutorialStep::SelectFaction => selection.faction.is_some(),
        TutorialStep::UseActions => selection.province.is_some(),
        TutorialStep::FollowChronicle => !game_state.events.is_empty(),
        TutorialStep::Complete => false,
    }
}

fn current_context(selection: &Selection, game_state: &ClientGameState) -> Option<String> {
    let world = game_state.world.as_ref()?;

    if let Some(pid) = selection.province {
        if let Some(province) = world.provinces.iter().find(|province| province.id == pid) {
            let owner = world
                .factions
                .iter()
                .find(|faction| faction.id == province.controller)
                .map(|faction| faction.name.as_str())
                .unwrap_or("Unknown");
            return Some(format!("Selected province: {} ({owner})", province.name));
        }
    }

    if let Some(fid) = selection.faction {
        if let Some(faction) = world.factions.iter().find(|faction| faction.id == fid) {
            return Some(format!("Selected faction: {}", faction.name));
        }
    }

    Some(format!(
        "World loaded: turn {}, {} factions, {} provinces",
        world.meta.turn,
        world.factions.len(),
        world.provinces.len()
    ))
}
