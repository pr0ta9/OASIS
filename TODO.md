TODO:
1. **Complete System Implementation** - Implement a fully functional multi-agent system that can handle complex tasks using existing tools
   - Ensure all agents (planning, text, image, audio, video, document) work seamlessly together
   - Test complex task workflows with current tool set
   - Optimize agent handoffs and communication
   - Validate system stability and reliability

2. **Tool Agent Implementation** - Add dynamic tool creation capability
   - Create a specialized "Tool Agent" that can write new tools on demand
   - Implement file creation system to dynamically generate `.py` files in appropriate tool directories
   - Add automatic import management and tool registration
   - Enable agents to request new tools when existing ones are insufficient
   - Example flow: User request → Planning Agent → Specialist Agent → Tool Agent (creates missing tool) → Back to Specialist Agent → Result

3. langgraph toolkit tested + bigtools(mongodb)
4. langraph connect to ui
5. ui improvement

future improvement:

- update toolkit everytime when started(ask permission)
- 