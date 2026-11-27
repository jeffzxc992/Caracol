import Carbon
import Foundation

private func caracolHotKeyHandler(_ callRef: EventHandlerCallRef?,
                                  _ event: EventRef?,
                                  _ userData: UnsafeMutableRawPointer?) -> OSStatus {
    guard let event, let userData else { return OSStatus(eventNotHandledErr) }
    var identifier = EventHotKeyID()
    let status = GetEventParameter(
        event,
        EventParamName(kEventParamDirectObject),
        EventParamType(typeEventHotKeyID),
        nil,
        MemoryLayout<EventHotKeyID>.size,
        nil,
        &identifier
    )
    guard status == noErr else { return status }
    let manager = Unmanaged<GlobalHotkey>.fromOpaque(userData).takeUnretainedValue()
    DispatchQueue.main.async {
        manager.onPress?()
    }
    return noErr
}

/// Global hotkey via Carbon. This needs no Accessibility permission or input
/// monitoring, unlike CGEvent taps. The default is Control-Option-Command-G.
final class GlobalHotkey {
    var onPress: (() -> Void)?
    private(set) var isRegistered = false

    private var hotKeyRef: EventHotKeyRef?
    private var eventHandler: EventHandlerRef?
    private let signature: OSType = 0x43415241 // 'CARA'

    deinit {
        unregister()
    }

    @discardableResult
    func register() -> Bool {
        guard !isRegistered else { return true }
        var eventType = EventTypeSpec(
            eventClass: OSType(kEventClassKeyboard),
            eventKind: UInt32(kEventHotKeyPressed)
        )
        let userData = Unmanaged.passUnretained(self).toOpaque()
        let installStatus = InstallEventHandler(
            GetApplicationEventTarget(),
            caracolHotKeyHandler,
            1,
            &eventType,
            userData,
            &eventHandler
        )
        guard installStatus == noErr else {
            LogStore.warn("hotkey", "InstallEventHandler failed: \(installStatus)")
            return false
        }

        let hotKeyID = EventHotKeyID(signature: signature, id: 1)
        let modifiers = UInt32(cmdKey | optionKey | controlKey)
        let registerStatus = RegisterEventHotKey(
            UInt32(kVK_ANSI_G),
            modifiers,
            hotKeyID,
            GetApplicationEventTarget(),
            0,
            &hotKeyRef
        )
        guard registerStatus == noErr, hotKeyRef != nil else {
            LogStore.warn("hotkey", "RegisterEventHotKey failed: \(registerStatus) (probably already in use)")
            if let eventHandler {
                RemoveEventHandler(eventHandler)
                self.eventHandler = nil
            }
            return false
        }
        isRegistered = true
        LogStore.info("hotkey", "Registered Control-Option-Command-G")
        _ = hotKeyID
        return true
    }

    func unregister() {
        if let hotKeyRef {
            UnregisterEventHotKey(hotKeyRef)
            self.hotKeyRef = nil
        }
        if let eventHandler {
            RemoveEventHandler(eventHandler)
            self.eventHandler = nil
        }
        isRegistered = false
    }
}
