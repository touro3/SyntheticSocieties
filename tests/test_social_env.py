from conftest import make_agent

from decision.social_policy import SocialAction, SocialPolicy
from environment.network import NetworkManager
from environment.social_env import SocialEnvironment
from environment.world_state import WorldState


def test_social_environment_post_comment_and_react_flow():
    agents = {aid: make_agent(aid) for aid in ["agent_0", "agent_1"]}
    env = SocialEnvironment(state=WorldState(round_id=1))

    post_action = SocialAction(action_type="post", content="A first community update.")
    validation = env.validate_action(post_action, agents["agent_0"], agents)
    assert validation.valid

    post_event = env.execute_action(post_action, agents["agent_0"], agents)
    assert post_event["post_id"] == "post_1"

    comment_action = SocialAction(action_type="comment", target_id="post_1", content="I can add local context.")
    comment_event = env.execute_action(comment_action, agents["agent_1"], agents)
    assert comment_event["parent_id"] == "post_1"

    react_action = SocialAction(action_type="react", target_id="post_1", reaction="upvote")
    env.execute_action(react_action, agents["agent_1"], agents)
    assert env.posts["post_1"].reactions["upvote"] == 1


def test_social_feed_prefers_network_neighbors():
    network = NetworkManager.fully_connected(["agent_0", "agent_1"])
    env = SocialEnvironment(state=WorldState(round_id=1), network_manager=network)
    env.submit_post("stranger", "Older public post.")
    env.submit_post("agent_1", "Neighbor update.")

    feed = env.get_feed("agent_0", n=2)

    assert feed[0].author_id == "agent_1"


def test_social_policy_parses_json_action():
    class Backend:
        def generate(self, messages, temperature=0.7):
            return (
                '{"action_type":"post","content":"hello","target_id":null,'
                '"reaction":null,"reasoning_summary":"say hello","confidence":0.9}',
                0.01,
            )

    policy = SocialPolicy(backend=Backend())
    agent = make_agent("agent_0")
    context = {"world": {"feed": [], "trending": [], "platform": "short_form"}, "network": {"neighbors": []}}

    action = policy.propose_action(agent.profile, agent.state, agent.memory, context, round_id=1)

    assert action.action_type == "post"
    assert action.content == "hello"
    assert action.model_dump()["target_agent_id"] is None
