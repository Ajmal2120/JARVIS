# JARVIS LLM Provider Migration Guide

This document explains the new provider-agnostic architecture and how to switch between Claude and Gemini.

## Overview

JARVIS now supports multiple LLM providers (Claude and Gemini) through a unified interface. This makes the application more resilient to API issues and allows easy switching between providers during development.

## Architecture

### New Structure

```
backend/
├── llm/
│   ├── __init__.py          # Public API (get_provider())
│   ├── provider.py          # Base LLMProvider interface
│   ├── claude_provider.py   # Claude implementation
│   ├── gemini_provider.py   # Gemini implementation
│   └── factory.py           # Provider factory + env detection
└── config.py                # Added LLM_PROVIDER setting
```

### Provider Interface

All providers implement the `LLMProvider` interface with these methods:

```python
@dataclass
class ChatResponse:
    text: str
    tool_calls: Optional[List[dict]] = None
    usage: Optional[dict] = None

class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list, **kwargs) -> ChatResponse:
        pass
    
    @abstractmethod
    async def close(self) -> None:
        pass
```

## Configuration

### Environment Variables

Add these to your `.env` file:

```env
# Set your preferred provider
LLM_PROVIDER=claude           # Options: claude, gemini

# Claude-specific settings (when LLM_PROVIDER=claude)
ANTHROPIC_API_KEY=your-claude-key
CLAUDE_MODEL=claude-sonnet-4-6

# Gemini-specific settings (when LLM_PROVIDER=gemini)
GEMINI_API_KEY=your-gemini-key
GEMINI_MODEL=gemini-2.0-flash-exp
```

### Default Provider

If `LLM_PROVIDER` is not set, it defaults to `claude`.

## Switching Providers

### To Use Claude (Default)

1. Set in `.env`:
```env
LLM_PROVIDER=claude
ANTHROPIC_API_KEY=your-claude-key
```

2. Make sure you have the required packages:
```bash
pip install anthropic>=0.39.0
```

### To Use Gemini

1. Set in `.env`:
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-key
```

2. Install required packages:
```bash
pip install google-generativeai>=0.8.0
```

### During Development

You can easily switch between providers by changing only the `LLM_PROVIDER` variable in your `.env` file and restarting the application.

## Migration Benefits

1. **Provider Agnostic**: Core code doesn't depend on specific LLM APIs
2. **Easy Switching**: Change providers by editing one line in `.env`
3. **Tool Support**: Both providers support the same tool definitions
4. **Backward Compatible**: Existing functionality is preserved
5. **Future Extensible**: Easy to add new providers (OpenAI, etc.)

## Behavior Notes

### Tool Use Loop

The tool execution pattern remains exactly the same:
1. First API call enables tools
2. Execute tools locally when requested
3. Send tool results back for follow-up generation

This ensures full compatibility with existing tools and workflows.

### Error Handling

If a provider fails:
- Graceful fallback to generic error message
- No provider-specific exceptions bubble up
- Logs the actual error for debugging

### Performance

- Provider initialization is lazy (on first call)
- Client instances are cached for reuse
- Connection pooling is provider-specific

## Known Limitations

1. **Vision Support**: Gemini's image format may differ from Claude
2. **Tool Definitions**: Some providers may have slight differences in tool schemas
3. **Streaming**: Not all providers support streaming (future enhancement)

## Next Steps

The architecture is designed to be extensible. Future work could include:
- Add OpenAI provider
- Implement response streaming
- Add rate limiting across providers
- Provider-specific tool configuration