# class TrimMessagesMiddleware(AgentMiddleware[CustomAgentState]):

#     state_schema = CustomAgentState

#     @before_model
#     def trim_messages(state: CustomAgentState, runtime: Runtime) -> dict[str, Any] | None:
#         """Keep only the last few messages to fit context window."""
#         messages = state["messages"]

#         if len(messages) <= 3:
#             return None  # No changes needed

#         first_msg = messages[0]
#         recent_messages = messages[-3:] if len(messages) % 2 == 0 else messages[-4:]
#         new_messages = [first_msg] + recent_messages

#         return {
#             "messages": [
#                 RemoveMessage(id=REMOVE_ALL_MESSAGES),
#                 *new_messages
#             ]
#         }


#     def delete_specfic_messages(state):
#         messages = state["messages"]
#         if len(messages) > 2:
#             # remove the earliest two messages
#             return {"messages": [RemoveMessage(id=m.id) for m in messages[:2]]}  
        

#     def delete_all_messages(state):
#         return {"messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES)]}  
