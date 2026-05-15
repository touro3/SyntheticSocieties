from environment.world_state import WorldState


class World:
    def __init__(self, state: WorldState, institution_manager, network_manager=None) -> None:
        self.state = state
        self.institution_manager = institution_manager
        self.network_manager = network_manager

    def get_agent_context(self, agent_id: str) -> dict:
        neighbors = []
        if self.network_manager is not None:
            neighbors = self.network_manager.get_neighbors(agent_id)

        return {
            "public_signal": self.state.public_signal,
            "prices": self.state.prices,
            "resources": self.state.resources,
            "neighbors": neighbors,
        }

    def validate_action(self, action, agent, agent_lookup):
        return self.institution_manager.validate(action, agent, self.state, agent_lookup)

    def execute_action(self, action, agent, agent_lookup) -> dict:
        return self.institution_manager.execute(action, agent, self.state, agent_lookup)

    def apply_exogenous_updates(self) -> list[dict]:
        self.state.round_id += 1
        applied = list(getattr(self.state, "pending_injections", []))
        self.state.pending_injections.clear()
        for injection in applied:
            self._apply_injection(injection)
        return applied

    def _apply_injection(self, injection: dict) -> None:
        if not isinstance(injection, dict):
            return
        event_type = str(injection.get("event_type", injection.get("type", "narrative")))
        payload = injection.get("payload", {})
        if not isinstance(payload, dict):
            payload = {"content": payload}

        if event_type == "wealth_shock":
            magnitude = payload.get("magnitude", payload.get("amount", payload.get("shock_magnitude", 0.0)))
            try:
                self.state.shock_magnitude = float(magnitude)
            except (TypeError, ValueError):
                self.state.shock_magnitude = 0.0
            self.state.shock_active = True
            message = payload.get("content") or payload.get("message") or f"wealth shock {self.state.shock_magnitude}"
            self.state.public_signal["wealth_shock"] = str(message)

        if event_type == "signal_update":
            signal = payload.get("signal", payload)
            if isinstance(signal, dict):
                self.state.public_signal.update({str(k): str(v) for k, v in signal.items()})

        if event_type == "scarcity":
            # Causal-diagnostic shock (Phase 3): broadcast a resource-scarcity
            # signal and scale the named resources down. Reuses the existing
            # pending_injections pathway — no kernel changes.
            self.state.public_signal["scarcity"] = "true"
            severity = payload.get("severity", payload.get("magnitude", 0.5))
            try:
                severity = float(severity)
            except (TypeError, ValueError):
                severity = 0.5
            message = payload.get("content") or payload.get("message") or f"resource scarcity (severity {severity})"
            self.state.public_signal["scarcity_message"] = str(message)
            targets = payload.get("resources")
            keys = targets if isinstance(targets, list) else list(self.state.resources.keys())
            for key in keys:
                if key in self.state.resources:
                    try:
                        self.state.resources[key] = float(self.state.resources[key]) * max(0.0, 1.0 - severity)
                    except (TypeError, ValueError):
                        continue

        if event_type == "narrative":
            content = payload.get("content", payload.get("message", ""))
            if content:
                self.state.public_signal["narrative"] = str(content)

        prices = payload.get("prices")
        if isinstance(prices, dict):
            for key, value in prices.items():
                try:
                    self.state.prices[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue

        resources = payload.get("resources")
        if isinstance(resources, dict):
            for key, value in resources.items():
                try:
                    self.state.resources[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
