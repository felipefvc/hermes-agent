from gateway.run import _transcript_has_session_meta


def test_transcript_meta_detection_handles_observed_only_history():
    assert not _transcript_has_session_meta([
        {"role": "user", "content": "https://youtu.be/siHfHUm3HGE", "observed": True},
    ])
    assert _transcript_has_session_meta([
        {"role": "user", "content": "https://youtu.be/siHfHUm3HGE", "observed": True},
        {"role": "session_meta", "tools": []},
    ])
