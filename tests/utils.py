from planning_through_contact.tools.types import NpExpressionArray


def assert_np_expression_array_eq(
    np1: NpExpressionArray, np2: NpExpressionArray
) -> None:
    for e1, e2 in zip(np1.flatten(), np2.flatten()):
        assert e1.EqualTo(e2)
